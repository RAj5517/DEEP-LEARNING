# Inference — Decoding Strategies & Post-Processing

---

## What Is Inference?

Training produces a model with learned parameters. Inference is how you **use that model to produce outputs** — and for generative models, the way you decode outputs from probability distributions dramatically affects quality, diversity, and behavior.

For discriminative models (classifiers, detectors), inference is straightforward: run the forward pass, apply post-processing. For **generative models** (language models, seq2seq), inference involves making decisions at each step about which token/word to generate next — and those decisions compound over the full output.

---

## Decoding Strategies (Generative Models)

These apply to any autoregressive model: language models (GPT), seq2seq (translation, summarization), speech synthesis, etc.

At each step, the model produces a **probability distribution over the vocabulary**. The decoding strategy decides which token to pick.

---

### Greedy Decoding

At each step, pick the **single highest-probability token**:

```python
output_ids = []
for _ in range(max_length):
    logits = model(input_ids)
    next_token = logits[:, -1, :].argmax(dim=-1)   # Always pick max
    output_ids.append(next_token)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
    if next_token == eos_token_id:
        break
```

**Advantage:** Fast, deterministic, no hyperparameters.

**Problem:** Locally optimal ≠ globally optimal. Greedy can commit to a word that looks good now but leads to a bad continuation. Prone to repetition loops ("the the the...").

**Use when:** Speed is critical, outputs are short, or the task has a well-defined single correct answer (code completion, math).

---

### Beam Search ⭐

Instead of one candidate, keep the **top-k sequences** (beams) at each step and expand all of them:

```
beam_size = 4

Step 1: 4 most likely first tokens               → 4 candidates
Step 2: expand each by 4 → top 4 of 16 total    → 4 candidates
Step 3: expand each by 4 → top 4 of 16 total    → 4 candidates
...final: return highest-scoring complete sequence
```

```python
# HuggingFace (beam search built-in):
outputs = model.generate(
    input_ids,
    num_beams=4,           # Number of beams
    num_return_sequences=1, # Return best sequence
    max_new_tokens=100,
    early_stopping=True    # Stop when all beams hit EOS
)
```

**Beam size tradeoffs:**

| beam_size | Quality | Speed | Diversity |
|-----------|---------|-------|-----------|
| 1 | = Greedy | Fastest | None |
| 4 | Good | Moderate | Low |
| 8–12 | Better | Slow | Low |
| >20 | Diminishing returns | Very slow | Very low |

**Problem:** Beam search still tends to produce **safe, generic, repetitive** text. The highest-probability sequence is often bland — "I am doing well. How can I help you?" for every conversation. It also penalizes diversity; all beams can collapse to near-identical sequences.

**Use when:** Tasks with a correct answer (translation, summarization where faithfulness matters), structured output (code, formulas).

---

### Sampling ⭐ Most Important for Open-Ended Generation

Instead of picking the max, **randomly sample** from the probability distribution. This introduces diversity and creativity.

**Pure random sampling** — sample directly from the full distribution:
```python
probs = torch.softmax(logits[:, -1, :], dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**Problem:** Can sample low-probability garbage tokens. "The president of France is... banana."

---

#### Temperature

Temperature scales the logits before softmax, controlling how **peaked or flat** the distribution is:

```python
logits = logits / temperature   # Scale before softmax
probs = torch.softmax(logits, dim=-1)
```

```
Temperature = 1.0  →  Original distribution (no change)
Temperature < 1.0  →  Sharper distribution, more confident, less diverse
Temperature > 1.0  →  Flatter distribution, more random, more creative
Temperature → 0    →  Approaches greedy (always picks the max)
Temperature → ∞    →  Uniform distribution (completely random)
```

**Practical values:**

| Task | Temperature |
|------|------------|
| Code generation | 0.2 – 0.4 |
| Factual Q&A | 0.3 – 0.7 |
| Creative writing | 0.7 – 1.0 |
| Brainstorming / diversity | 0.9 – 1.2 |

---

#### Top-k Sampling

Before sampling, **keep only the top-k most probable tokens** and redistribute probability mass among them:

```python
def top_k_sampling(logits, k=50):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    # Set all other logits to -inf so they get 0 probability
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(1, top_k_indices, top_k_logits)
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Problem:** The right `k` depends on context. Sometimes 50 candidates are all reasonable; sometimes only 5 are. A fixed `k` is arbitrary.

---

#### Top-p Sampling (Nucleus Sampling) ⭐

Instead of a fixed number of candidates, keep the **smallest set of tokens whose cumulative probability exceeds p**:

```python
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above p
    sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    logits.scatter_(1, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

When the model is confident (one token dominates), the nucleus is small — only a few tokens are sampled from. When the model is uncertain (many tokens have similar probability), the nucleus is large — more diversity is allowed. This adapts dynamically to context.

**Top-p = 0.9** is the most common default. At p=0.9, you keep the smallest set of tokens that account for 90% of probability mass.

---

#### Combining Everything: The Standard Recipe

```python
outputs = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,    # Penalize repeating tokens
    max_new_tokens=200,
)
```

Using temperature + top-p together is the modern standard for open-ended generation. Top-k is sometimes added as an additional filter. `repetition_penalty > 1.0` discourages repeating tokens that have already appeared.

---

### Decoding Strategy Decision Guide

```
Is there a single correct answer? (math, code, factual Q&A)
  └─ YES → Greedy or Beam Search (num_beams=4)

Do you need faithful/accurate output? (translation, summarization)
  └─ YES → Beam Search (num_beams=4–8)

Do you want creative/diverse output? (chat, creative writing, story)
  └─ YES → Sampling with temperature=0.7–1.0 + top_p=0.9

Do you need multiple diverse outputs?
  └─ YES → Beam Search with num_return_sequences=N + diversity_penalty
         → Or: run sampling multiple times
```

---

## Post-Processing

After the model produces raw outputs, post-processing converts them into final usable predictions.

---

### Non-Maximum Suppression (NMS) — Object Detection ⭐

Object detectors produce many overlapping bounding boxes for the same object. NMS eliminates redundant boxes, keeping only the best one per object:

```
Algorithm:
1. Sort all predicted boxes by confidence score (high → low)
2. Take the highest-confidence box → keep it
3. Remove all other boxes with IoU > threshold (e.g., 0.5) with the kept box
4. Repeat with the next highest-confidence remaining box
5. Continue until no boxes remain
```

```python
import torchvision

# boxes: [N, 4] tensor of [x1, y1, x2, y2]
# scores: [N] confidence scores
keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
final_boxes = boxes[keep_indices]
final_scores = scores[keep_indices]
```

**IoU threshold tradeoffs:**
- Low threshold (0.3): aggressive — removes boxes that slightly overlap → may miss nearby same-class objects
- High threshold (0.7): permissive — keeps more overlapping boxes → may have duplicates

**Soft-NMS** — instead of hard removal, reduces the score of overlapping boxes. Better for dense scenes where objects legitimately overlap (crowds, shelves):

```python
keep_indices, updated_scores = torchvision.ops.soft_nms(
    boxes, scores, iou_threshold=0.5, sigma=0.5
)
```

---

### Calibration — Making Probabilities Trustworthy

A model's raw output probabilities are often **miscalibrated** — a softmax output of 0.9 doesn't necessarily mean 90% confidence in the real world. Overconfidence is common, especially in models trained with hard labels.

**Temperature Scaling** — the simplest and most effective calibration method. Learns a single scalar `T` applied to all logits before softmax:

```python
# After training, on validation set:
# Learn T that minimizes NLL of calibrated predictions

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature

# Calibrate using LBFGS on validation set logits:
calibrator = TemperatureScaling()
optimizer = torch.optim.LBFGS([calibrator.temperature], lr=0.01, max_iter=50)

def eval_loss():
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(calibrator(val_logits), val_labels)
    loss.backward()
    return loss

optimizer.step(eval_loss)
# Now use calibrator(logits) instead of raw logits at test time
```

**Reliability diagram** — visual check for calibration: bin predictions by confidence, plot actual accuracy per bin. A calibrated model's curve lies on the diagonal.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
plt.plot(prob_pred, prob_true)
plt.plot([0,1], [0,1], 'k--')   # Perfect calibration line
```

---

### Thresholding — Controlling Precision/Recall Tradeoff

For binary classification and detection, the default threshold of 0.5 is rarely optimal. The right threshold depends on your deployment requirements:

**Finding the optimal threshold:**

```python
from sklearn.metrics import precision_recall_curve, roc_curve

# Precision-Recall based (when FN vs FP cost matters)
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Find threshold that maximizes F1
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
optimal_threshold = thresholds[f1_scores.argmax()]

# Or set a minimum recall requirement (e.g., medical: recall ≥ 0.95)
min_recall = 0.95
valid_thresholds = thresholds[recall[:-1] >= min_recall]
optimal_threshold = valid_thresholds.max()   # Highest threshold that still meets recall
```

**Business-driven thresholds:**

| Scenario | Strategy |
|----------|----------|
| Equal cost for FP and FN | Maximize F1 |
| FN is very costly (medical screening) | Set minimum recall = 0.95–0.99 |
| FP is very costly (spam filtering) | Set minimum precision = 0.90–0.99 |
| Need to handle a quota of positives | Set threshold to control predicted positive rate |

Once you have a threshold from validation, apply it consistently at test time:

```python
final_predictions = (y_scores >= optimal_threshold).astype(int)
```

---

## Inference Optimization (Brief)

Not decoding strategies, but important for production:

**`torch.no_grad()`** — always wrap inference in this. Disables gradient computation, saves ~50% memory and speeds up inference:
```python
with torch.no_grad():
    outputs = model(inputs)
```

**`model.eval()`** — disables dropout and sets BatchNorm to use running statistics (not batch statistics). Always call before inference.

**Half precision (fp16/bf16)** — halves memory, speeds up inference on modern GPUs:
```python
model = model.half()   # fp16
inputs = inputs.half()
```

**torch.compile** (PyTorch 2.0+) — JIT compilation for significant speedups:
```python
model = torch.compile(model)
```

---

## Quick Reference

| Strategy | Best For | Key Parameter |
|----------|---------|---------------|
| Greedy | Fast, deterministic, correct-answer tasks | — |
| Beam Search | Translation, summarization, structured output | `num_beams=4` |
| Temperature Sampling | Creative, open-ended generation | `temp=0.7–1.0` |
| Top-p (Nucleus) | Open-ended + quality control | `top_p=0.9` |
| NMS | Object detection box filtering | `iou_threshold=0.5` |
| Temperature Scaling | Calibrating probabilities | Fit `T` on val set |
| Thresholding | Controlling precision/recall | Fit on val set by objective |