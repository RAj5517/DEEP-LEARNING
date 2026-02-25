# Label Smoothing — Regularization for Classification Targets

---

## What Is Label Smoothing?

In a standard classification setup, targets are **one-hot encoded**: the true class gets probability 1.0 and all other classes get 0.0. This seems natural, but it creates a specific problem: the model is trained to be **infinitely confident** about its predictions.

Cross-entropy loss with one-hot targets has no finite minimum — you can always reduce the loss by making the logit for the true class larger. The model is incentivized to push the correct class score toward `+∞` and all other class scores toward `-∞`, which means logit values grow without bound during training.

This leads to:
1. **Overconfident predictions** — probabilities like `0.9999` on the correct class, even for ambiguous or mislabeled examples
2. **Poor calibration** — the model's confidence scores don't reflect actual accuracy (a model that's 99% confident should be right ~99% of the time; overfit models often aren't)
3. **Reduced generalization** — the model adapts to the exact training labels rather than the underlying distribution

**Label smoothing** (Szegedy et al., 2015, introduced in Inception-v3) fixes this by replacing the hard `[0, ..., 1, ..., 0]` target distribution with a **soft distribution** that assigns a small probability `ε/(K-1)` to all non-target classes and `1 - ε` to the true class:

```
# Hard label (standard):
y_hard = [0, 0, 1, 0, 0]   # One-hot for class 2 out of 5

# Smoothed label (with ε = 0.1):
y_smooth = [0.02, 0.02, 0.92, 0.02, 0.02]
# true class: 1 - ε = 0.90, others: ε/(K-1) = 0.1/4 = 0.025
```

Now the model is penalized for being too confident: it can never achieve zero loss no matter how large the correct class logit becomes. The loss has a finite minimum at a specific level of confidence, not at infinite confidence.

---

## The Math

### Standard Cross-Entropy Loss

```
L_CE = -Σ y_k · log(p_k)

# With one-hot y: simplifies to
L_CE = -log(p_true)
```

### Label-Smoothed Cross-Entropy Loss

```
# Smoothed target:
y_smooth_k = (1 - ε) · y_k + ε / K

# Loss:
L_LS = -Σ y_smooth_k · log(p_k)
     = (1 - ε) · (-log(p_true)) + ε · (-1/K · Σ log(p_k))
     = (1 - ε) · L_CE + ε · H_uniform
```

Where `H_uniform = -1/K · Σ log(p_k)` is the cross-entropy against the uniform distribution.

**Interpretation:** Label smoothing adds a term that penalizes the model for being **too far from the uniform distribution** — it adds a "stay humble" penalty that prevents the logits from growing without bound.

The gradient tells the story clearly. With standard cross-entropy, the gradient with respect to the true class logit is:
```
∂L_CE/∂z_true = p_true - 1
```
This is always negative (the gradient always pushes the true class logit higher). It only reaches zero at `p_true = 1`, which requires `z_true → ∞`.

With label smoothing:
```
∂L_LS/∂z_true = p_true - (1 - ε)
```
This reaches zero at `p_true = 1 - ε`. For `ε = 0.1`, the gradient stops pushing at `p_true = 0.9`. **The model has a finite optimal confidence level, not infinite.**

---

## Choosing ε: The Smoothing Factor

The standard value is `ε = 0.1`. This is used in Inception-v3, BERT, most ViT models, and a wide range of language models.

| ε value | Effect |
|---------|--------|
| 0.0 | Standard cross-entropy, no smoothing |
| 0.05 | Mild smoothing — subtle regularization |
| **0.1** | **Standard — the default in almost all papers** |
| 0.2 | Aggressive smoothing — use for heavily noisy labels |
| 0.3+ | Usually too aggressive — hurts accuracy |

**Larger ε = more regularization, but also less useful gradient signal from labels.** With very large ε, the model is spending too much effort predicting non-target classes and loses accuracy.

For **noisy label** settings (e.g., web-scraped data where some labels are wrong), higher ε (0.15–0.2) is appropriate because the labels themselves are uncertain.

---

## Implementation

### PyTorch Built-in

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Usage in training loop:
logits = model(images)     # Shape: [batch, num_classes]
loss = loss_fn(logits, labels)   # labels: integer class indices
```

PyTorch's built-in handles everything — just pass `label_smoothing=0.1` and you're done. This is the recommended approach.

### Manual Implementation (for Understanding)

```python
def label_smoothed_cross_entropy(logits, targets, epsilon=0.1, num_classes=1000):
    # Create soft labels
    batch_size = logits.size(0)
    
    # Start with one-hot
    smooth_labels = torch.zeros_like(logits).scatter_(
        1, targets.unsqueeze(1), 1.0
    )
    
    # Apply smoothing
    smooth_labels = smooth_labels * (1 - epsilon) + epsilon / num_classes
    
    # Cross-entropy loss with soft labels
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
    
    return loss
```

### In Hugging Face (for NLP)

```python
# In transformers, label smoothing is often built into the Trainer:
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    label_smoothing_factor=0.1,   # Built directly into Trainer
    ...
)
```

---

## What Label Smoothing Actually Does to the Model

### Better Calibration

**Calibration** measures whether a model's confidence scores match its actual accuracy. A well-calibrated model that outputs 80% confidence should be correct about 80% of the time.

Without label smoothing, models tend to be **overconfident**: they output 99% confidence on many examples but are only actually correct 90% of the time. Label smoothing directly constrains the maximum confidence level, producing better-calibrated probability outputs.

This is critically important in production systems where you use the model's confidence as a decision threshold (e.g., flagging uncertain predictions for human review).

### Representation Quality

An interesting finding (Müller et al., 2019 — "When Does Label Smoothing Help?"): label smoothing produces **tighter, more clustered penultimate-layer representations**. Without label smoothing, the representation of class `k` is pulled toward a single "template" direction. With smoothing, representations cluster more tightly and are more separated between classes.

This is why models trained with label smoothing often make **better feature extractors for transfer learning** — their representations are more structured and generalizable.

### The Downside: Knowledge Distillation

The same paper showed that label-smoothed models make **worse teachers** in knowledge distillation. The soft logits from label-smoothed models contain less useful information about class relationships (the "dark knowledge") because the logits are compressed toward more uniform distributions. If you plan to use a model as a teacher for distillation, either skip label smoothing or use a lower ε.

---

## When to Use Label Smoothing

**Use label smoothing when:**
- Training classification models from scratch on large datasets (ImageNet, large NLP corpora)
- Your model outputs will be used as confidence scores / probabilities (calibration matters)
- You have potentially noisy labels (web data, crowd-sourced annotations)
- Training transformers — essentially all modern transformer training uses it

**Be cautious when:**
- Using the model as a **teacher for knowledge distillation** — label smoothing hurts the quality of soft targets
- Very small datasets where every bit of label signal matters
- Regression tasks, segmentation with pixelwise loss — label smoothing is primarily for classification

**Don't use when:**
- You need maximum predictive accuracy and calibration doesn't matter
- Sequential prediction with cross-entropy where you want sharp next-token distributions (though many LLMs still use it)

---

## Label Smoothing vs. Other Regularizers

| Regularizer | Mechanism | Effect |
|-------------|-----------|--------|
| Weight Decay | Shrinks parameters | Limits model capacity |
| Dropout | Drops activations | Forces redundant representations |
| Early Stopping | Limits training time | Limits effective capacity |
| **Label Smoothing** | Softens targets | Limits confidence, improves calibration |
| Mixup | Interpolates samples | Encourages linear behavior |

Label smoothing is **complementary to all others** — it operates on the target distribution rather than the model architecture or optimizer. Using label smoothing alongside dropout and weight decay is standard practice.

---

## Key Takeaways

- Label smoothing is a **one-line change** (`nn.CrossEntropyLoss(label_smoothing=0.1)`) with consistent, meaningful gains — there's rarely a reason not to use it for classification.
- The standard value is **ε = 0.1** across the board — this rarely needs tuning.
- It prevents models from becoming **overconfident** by giving logit growth a finite optimum.
- Label smoothing significantly improves **calibration** — particularly important in production systems using confidence thresholds.
- The one important exception: **don't use it for knowledge distillation** — smoothed teachers provide worse soft targets.
- It produces **better penultimate-layer representations**, making the model a better feature extractor for downstream tasks.