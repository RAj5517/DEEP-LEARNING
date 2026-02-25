# Evaluation Metrics

---

## Why Metrics Matter

Accuracy alone will mislead you. A model that predicts "not fraud" for every transaction on a dataset with 99% non-fraud will be 99% accurate — and completely useless. Every task requires metrics that reflect what you actually care about. Choosing the wrong metric means optimizing for the wrong thing.

---

## Classification Metrics

### The Confusion Matrix — Foundation of Everything

Before any metric, understand the four outcomes:

```
                    Predicted
                  Positive  Negative
Actual  Positive │   TP   │   FN   │
        Negative │   FP   │   TN   │
```

- **TP** (True Positive): correctly predicted positive
- **TN** (True Negative): correctly predicted negative
- **FP** (False Positive): predicted positive, actually negative — "false alarm"
- **FN** (False Negative): predicted negative, actually positive — "missed detection"

All classification metrics are derived from these four numbers.

---

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use:** Balanced classes, and all errors cost roughly the same.

**When NOT to use:** Imbalanced datasets. A 99% negative dataset gives 99% accuracy by always predicting negative. Useless.

---

### Precision & Recall ⭐

These two are the real workhorses for imbalanced problems.

```
Precision = TP / (TP + FP)    "Of everything I predicted positive, how many were actually positive?"
Recall    = TP / (TP + FN)    "Of everything actually positive, how many did I find?"
```

They are always in tension — increasing one typically decreases the other. The trade-off is controlled by the **classification threshold** (default 0.5).

**Precision matters when:** False positives are costly. Email spam detection — you don't want to mark real emails as spam.

**Recall matters when:** False negatives are costly. Cancer screening — you don't want to miss a positive case.

```python
from sklearn.metrics import precision_score, recall_score, classification_report

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print(classification_report(y_true, y_pred))  # Shows both for all classes
```

---

### F1 Score ⭐

The harmonic mean of precision and recall. Balances both into a single number:

```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

The harmonic mean (not arithmetic mean) is used because it punishes extreme imbalances — a model with precision=1.0 and recall=0.01 gets F1=0.02, not 0.505.

**F1 is the go-to metric for imbalanced classification.** It's what you report when classes are skewed and both precision and recall matter.

**F-beta** — when you want to weight recall over precision:
```
F_β = (1 + β²) · (Precision · Recall) / (β² · Precision + Recall)

β=0.5 → weighs precision more
β=2   → weighs recall more (common in medical settings)
```

**Multi-class F1** — three averaging strategies:
- `macro`: average F1 per class equally (treats minority classes equally)
- `weighted`: average F1 weighted by class frequency
- `micro`: compute globally across all TP/FP/FN

For imbalanced multi-class → use `macro` to not let majority class dominate.

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='macro')
```

---

### ROC-AUC ⭐

ROC (Receiver Operating Characteristic) curve plots **True Positive Rate (Recall) vs False Positive Rate** at every possible threshold. AUC (Area Under Curve) summarizes this into a single number between 0 and 1.

```
AUC = 0.5  →  Random classifier
AUC = 1.0  →  Perfect classifier
AUC = 0.7–0.8  →  Decent
AUC = 0.8–0.9  →  Good
AUC > 0.9  →  Excellent
```

**What AUC measures:** The probability that the model ranks a random positive example higher than a random negative example. It evaluates the **ranking quality** of the model's scores, independent of threshold choice.

**Use AUC when:** You want a threshold-independent metric, or you'll be selecting a threshold post-hoc for deployment.

**PR-AUC (Precision-Recall AUC)** — more informative than ROC-AUC for highly imbalanced datasets. ROC-AUC can be optimistic when negatives vastly outnumber positives.

```python
from sklearn.metrics import roc_auc_score, average_precision_score

roc_auc = roc_auc_score(y_true, y_scores)       # ROC-AUC
pr_auc  = average_precision_score(y_true, y_scores)  # PR-AUC (better for imbalance)
```

---

### Which Classification Metric to Use?

| Situation | Metric |
|-----------|--------|
| Balanced classes, equal error costs | Accuracy |
| Imbalanced, need one number | F1 (macro) |
| FP is costly (spam, alerts) | Precision |
| FN is costly (medical, fraud) | Recall |
| Threshold will be tuned later | ROC-AUC |
| Severe imbalance (1:100+) | PR-AUC |

---

## Regression Metrics

### MAE — Mean Absolute Error

```
MAE = (1/n) · Σ |y_true - y_pred|
```

Average absolute error. **Robust to outliers** — a prediction off by 100 contributes exactly 100 to MAE, not 10,000.

**Interpretable in the original units:** If predicting house prices in dollars, MAE = $15,000 means your average prediction is $15k off.

---

### MSE & RMSE — Mean Squared Error

```
MSE  = (1/n) · Σ (y_true - y_pred)²
RMSE = √MSE
```

Squares the errors, making **large errors disproportionately costly**. An error of 10 contributes 100; an error of 100 contributes 10,000. Use when large errors are significantly worse than small ones.

**RMSE** is in the same units as the target, making it more interpretable than MSE. **Report RMSE, optimize MSE** (they're equivalent for optimization since √ is monotonic).

**MAE vs RMSE:** If your data has outliers you care about → RMSE. If outliers are noise you want to be robust against → MAE.

---

### R² — Coefficient of Determination

```
R² = 1 - (SS_residual / SS_total)
   = 1 - Σ(y_true - y_pred)² / Σ(y_true - ȳ)²
```

Measures what fraction of variance in the target is explained by the model.

```
R² = 1.0  →  Perfect predictions
R² = 0.0  →  Model does no better than predicting the mean
R² < 0    →  Model is worse than just predicting the mean
```

**Useful for comparing models across different datasets** because it's scale-independent. Not useful in isolation — a high R² can still mean terrible absolute error.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_true, y_pred)
```

---

## Object Detection Metrics ⭐

### IoU — Intersection over Union

The foundation of all detection metrics:

```
IoU = Area of Overlap / Area of Union
```

Measures how well a predicted bounding box overlaps the ground truth box:

```
IoU = 0   →  No overlap
IoU = 1   →  Perfect overlap
IoU = 0.5 →  Common threshold for "correct" detection
```

```python
def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union
```

A predicted box is a **True Positive** if IoU ≥ threshold (usually 0.5), else False Positive.

---

### AP and mAP ⭐

**AP (Average Precision)** per class: the area under the Precision-Recall curve, computed by varying the confidence threshold.

**mAP (mean Average Precision)**: average AP across all classes.

```
mAP = (1/num_classes) · Σ AP_class
```

**mAP@0.50** — computes AP at IoU threshold 0.5 (a detection is correct if IoU ≥ 0.5). The PASCAL VOC standard.

**mAP@0.50:0.95** — computes AP averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05. The COCO standard. Much harder — it rewards precise box localization, not just rough detection.

```
mAP@.50:.95 = (1/10) · (mAP@0.50 + mAP@0.55 + ... + mAP@0.95)
```

**Which to use:** COCO (mAP@.50:.95) is the modern standard. PASCAL VOC (mAP@.50) is still reported for comparison with older papers.

```python
# Using torchmetrics
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(iou_type="bbox")
metric.update(preds, targets)
result = metric.compute()
# result['map']       → mAP@.50:.95
# result['map_50']    → mAP@.50
```

---

## Segmentation Metrics ⭐

### Pixel Accuracy

```
Pixel Accuracy = Correctly classified pixels / Total pixels
```

Same problem as classification accuracy — dominated by background class. A model that predicts "background" everywhere on a dataset with 90% background gets 90% accuracy. Almost never reported alone.

---

### IoU (per class) and Mean IoU ⭐

Segmentation IoU applies the same overlap concept to **segmented regions** instead of bounding boxes:

```
IoU_class = TP_pixels / (TP_pixels + FP_pixels + FN_pixels)
mIoU = (1/num_classes) · Σ IoU_class
```

**mIoU is the standard metric for semantic segmentation.** It evaluates each class separately and averages, so minority classes aren't hidden by dominant ones.

---

### Dice Coefficient (F1 for Segmentation) ⭐

```
Dice = 2·TP / (2·TP + FP + FN)
     = 2 · |Pred ∩ GT| / (|Pred| + |GT|)
```

Mathematically equivalent to F1 score, applied to pixel masks. Commonly used in **medical image segmentation** (brain tumors, organ segmentation). Often used as the training loss (`1 - Dice`) as well as the evaluation metric.

```
Dice = 0  →  No overlap
Dice = 1  →  Perfect segmentation
```

**Dice vs IoU:**
```
Dice = 2·IoU / (1 + IoU)   ← Dice is always ≥ IoU for a given prediction
```

Both are valid — Dice is slightly more forgiving of small overlaps. Medical imaging tends to use Dice; autonomous driving/scene understanding tends to use mIoU.

```python
def dice_coefficient(pred_mask, true_mask, smooth=1e-6):
    pred_flat = pred_mask.view(-1).float()
    true_flat = true_mask.view(-1).float()
    intersection = (pred_flat * true_flat).sum()
    return (2 * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)
```

---

## Language Model Metrics

### Perplexity

```
Perplexity = exp(average negative log-likelihood per token)
           = exp(-(1/N) · Σ log P(token_i | context))
```

Measures how surprised the model is by the test text. Lower is better.

```
Perplexity = 1    →  Perfect (model assigns probability 1 to every token)
Perplexity = 10   →  Model is choosing between ~10 equally likely tokens at each step
Perplexity = 100  →  Model is confused
```

**Used for:** Evaluating language model quality (GPT, BERT MLM head). Not interpretable across different tokenizers — a model with a larger vocabulary will have higher perplexity. Only compare perplexity across models with the same tokenizer and test set.

---

### BLEU ⭐ (Translation / Generation)

BLEU (Bilingual Evaluation Understudy) compares generated text to one or more reference translations by measuring **n-gram overlap**.

```
BLEU = BP · exp(Σ wₙ · log pₙ)

pₙ = precision of n-gram matches (n=1,2,3,4)
BP = brevity penalty (penalizes short generations)
```

Scores range 0–100. Generally: >30 is understandable, >50 is good quality, >60 is excellent.

**Limitations:** BLEU penalizes valid paraphrases, is insensitive to word order within matched n-grams, and correlates poorly with human judgment on some tasks. Despite this, it remains the most reported metric for machine translation.

```python
from nltk.translate.bleu_score import corpus_bleu
# references: list of list of reference token lists
# hypotheses: list of token lists
score = corpus_bleu(references, hypotheses)
```

---

### ROUGE ⭐ (Summarization)

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram overlap between generated summary and reference, **focused on recall** (how much of the reference is covered).

Key variants:
- **ROUGE-1**: unigram overlap (individual words)
- **ROUGE-2**: bigram overlap (word pairs)
- **ROUGE-L**: longest common subsequence (accounts for word order)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(reference, hypothesis)
# scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, etc.
```

**ROUGE-L is often most informative** — it captures sentence structure, not just word bag overlap.

---

## Generation Quality Metrics

### FID — Fréchet Inception Distance (Images)

Measures quality of **generated images** by comparing the statistical distribution of generated images to real images in feature space (using an Inception network's intermediate activations).

```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2·√(Σ_real·Σ_gen))
```

**Lower FID = better.** A model generating images nearly indistinguishable from real ones will have low FID.

```
FID ≈ 0     →  Generated images statistically identical to real
FID < 10    →  Excellent quality (state-of-the-art diffusion models)
FID < 30    →  Good quality
FID > 100   →  Noticeable artifacts or diversity issues
```

FID requires a **large sample** (typically 50k images) to be reliable — small samples give noisy estimates.

---

### MOS — Mean Opinion Score (Audio)

A human evaluation metric for audio quality: listeners rate generated audio on a 1–5 scale, then scores are averaged.

```
1 = Bad, 2 = Poor, 3 = Fair, 4 = Good, 5 = Excellent
```

No automated substitute fully replaces MOS for audio — automated metrics like PESQ or STOI exist but correlate imperfectly with human perception.

---

## Ranking Metrics ⭐

Used in search engines, recommender systems, and information retrieval.

### NDCG — Normalized Discounted Cumulative Gain

Measures ranking quality when results have **graded relevance** (not just relevant/not-relevant). Gives more credit for placing highly relevant items at the top.

```
DCG@k = Σᵢ₌₁ᵏ  relevance_i / log₂(i + 1)

NDCG@k = DCG@k / IDCG@k     (IDCG = ideal DCG with perfect ranking)
```

NDCG = 1.0 means perfect ranking. Items at position 1 are discounted by `log₂(2) = 1` (full credit), items at position 2 by `log₂(3) ≈ 1.58` (less credit), and so on.

**Use NDCG when:** Items have multiple relevance levels (e.g., 0=irrelevant, 1=somewhat relevant, 2=highly relevant). Standard for search and recommendation benchmarks.

---

### MRR — Mean Reciprocal Rank

For tasks where there's exactly **one correct answer**: measures where in the ranked list the first correct answer appears.

```
MRR = (1/|Q|) · Σ 1/rank_i
```

If the correct answer is at rank 1 → score 1.0. Rank 2 → 0.5. Rank 5 → 0.2.

**Use MRR when:** Exactly one correct answer exists and you care about finding it quickly. Q&A retrieval, entity linking, one-answer search.

```python
def mrr(ranked_lists, correct_answers):
    reciprocal_ranks = []
    for ranked, correct in zip(ranked_lists, correct_answers):
        for rank, item in enumerate(ranked, start=1):
            if item == correct:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

**MRR vs NDCG:** MRR assumes one correct answer and only cares about its position. NDCG handles graded relevance across multiple results. In practice: MRR for Q&A and simple retrieval, NDCG for search and recommendation.

---

## Quick Metric Selection Guide

| Task | Primary Metric | Secondary |
|------|---------------|-----------|
| Binary classification (balanced) | Accuracy | F1 |
| Binary classification (imbalanced) | F1 / PR-AUC | ROC-AUC |
| Multi-class classification | macro F1 | Accuracy |
| Regression (outliers matter) | RMSE | R² |
| Regression (robust to outliers) | MAE | R² |
| Object detection | mAP@.50:.95 | mAP@.50 |
| Semantic segmentation | mIoU | Dice |
| Medical segmentation | Dice | IoU |
| Machine translation | BLEU | — |
| Summarization | ROUGE-L | ROUGE-1/2 |
| Language modeling | Perplexity | — |
| Image generation | FID | — |
| Audio generation | MOS | PESQ |
| Search / recommendation | NDCG@k | MRR |
| Q&A retrieval | MRR | NDCG |