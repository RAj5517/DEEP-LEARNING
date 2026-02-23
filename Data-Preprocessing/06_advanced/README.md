# ⚙️ Advanced Preprocessing — Feature Engineering, Selection & Reduction

> More data doesn't mean better models.
> Better features do.
> This module covers everything between raw high-dimensional data and a model-ready input.

---

## 🔁 Pipeline at a Glance

```
High-Dimensional / Complex Data
   ↓
Feature Engineering      (create better signal from raw data)
   ↓
Feature Selection        (remove noise, redundancy, weak features)
   ↓
Dimensionality Reduction (compress into dense, meaningful space)
   ↓
Model
```

---

## 📂 Structure

| File | Covers |
|---|---|
| `01_feature_engineering.md` | Polynomial, binning, ratios, datetime, lag, aggregation, domain features |
| `02_feature_selection.md` | Variance filter, correlation, mutual info, RFE, SHAP, Lasso |
| `03_dimensionality_reduction.md` | PCA, Kernel PCA, UMAP, t-SNE, Autoencoder, VAE |

---

## ❓ Why This Matters

| Problem | Caused By |
|---|---|
| Model can't learn time patterns | Raw timestamp instead of hour/day/month features |
| Overfitting on wide data | No feature selection — model memorizes noise |
| Distance models fail | Curse of dimensionality — no reduction |
| Slow training | Too many features — most adding noise not signal |
| PCA destroys structure | Data is non-linear — use UMAP or Autoencoder instead |

---

## ⚡ When to Apply Each Step

| Step | Always | Only When |
|---|---|---|
| Feature Engineering | ✅ Tabular / time series | Domain knowledge available |
| Variance Filter | ✅ | After scaling |
| Correlation Filter | ✅ | Before model-based selection |
| Mutual Info / RFE | ⚠️ | > 50 features |
| SHAP Selection | ⚠️ | After first model is trained |
| PCA | ⚠️ | Linear structure · correlated features |
| Autoencoder | ⚠️ | Non-linear · enough training data |
| t-SNE / UMAP | ⚠️ | Visualization only (t-SNE) or preprocessing (UMAP) |

---

## 🔬 Core Idea

Three distinct problems, three distinct solutions:

**Engineering** — create signal that doesn't exist in the raw data.
**Selection** — eliminate features that hurt more than they help.
**Reduction** — compress many features into fewer without losing structure.

Most model failures in industry are feature failures — not architecture failures.

---

*For deep breakdowns, math, and code — refer to the individual files above.*