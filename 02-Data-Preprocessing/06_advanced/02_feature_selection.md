# Feature Selection

Feature selection removes features that are redundant, irrelevant, or harmful to model performance. More features is not always better — irrelevant features add noise, increase training time, and can cause overfitting.

> "Every irrelevant feature you keep is a way for the model to overfit."

---

## 1. Why Remove Features?

| Problem | Caused By |
|---|---|
| Overfitting | Noisy / irrelevant features give the model false patterns |
| Slow training | More features = more parameters = more computation |
| Multicollinearity | Correlated features confuse linear models and distort coefficients |
| Curse of dimensionality | In high dimensions, all points become equidistant — distance-based models fail |
| Memory cost | Wide feature matrices are expensive to store and process |

---

## 2. Remove Low-Variance Features

Features with near-zero variance carry almost no information — they are nearly constant across all samples.

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with variance < threshold
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)

# See which features were kept
kept_features = X.columns[selector.get_support()]
removed = set(X.columns) - set(kept_features)
print(f"Removed {len(removed)} low-variance features: {removed}")
```

**Note:** Apply after scaling — variance is scale-dependent. A feature ranging [0,1] will have much lower variance than one ranging [0,1000].

---

## 3. Remove Highly Correlated Features

Two highly correlated features carry the same information. Keep one, drop the other.

```python
import pandas as pd
import numpy as np

def drop_correlated(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Dropping {len(to_drop)} correlated features: {to_drop}")
    return df.drop(columns=to_drop)

df_reduced = drop_correlated(df, threshold=0.95)
```

---

## 4. Filter Methods (Statistical Tests)

Rank features by their statistical relationship with the target — independent of any model.

### For Classification

```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

# Chi-squared test (non-negative features only)
selector = SelectKBest(chi2, k=20)
X_selected = selector.fit_transform(X, y)

# ANOVA F-test (linear relationship)
selector = SelectKBest(f_classif, k=20)

# Mutual Information (non-linear, captures any relationship)
selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)

# View scores
scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
print(scores.head(20))
```

### For Regression

```python
from sklearn.feature_selection import f_regression, mutual_info_regression

selector = SelectKBest(mutual_info_regression, k=20)
X_selected = selector.fit_transform(X, y)
```

---

## 5. Wrapper Methods

Train a model repeatedly with different feature subsets and evaluate performance. More accurate than filter methods but computationally expensive.

### Recursive Feature Elimination (RFE)

Trains a model, ranks features by importance, removes the weakest, repeats.

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# RFE with fixed number of features
rfe = RFE(
    estimator  = RandomForestClassifier(n_estimators=100),
    n_features_to_select = 20,
    step = 1    # remove 1 feature per iteration
)
rfe.fit(X_train, y_train)
X_selected = rfe.transform(X_train)

# RFECV — automatically finds optimal number via cross-validation
rfecv = RFECV(
    estimator = RandomForestClassifier(n_estimators=100),
    cv = 5,
    scoring = "roc_auc"
)
rfecv.fit(X_train, y_train)
print(f"Optimal number of features: {rfecv.n_features_}")
```

---

## 6. Embedded Methods (Feature Importance from Models)

The model itself identifies which features matter during training.

### Tree-Based Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

# Select top N features
top_features = importance.head(20).index.tolist()
X_selected = X[top_features]

# Plot
importance.head(20).plot(kind="bar", figsize=(12, 4))
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()
```

### SHAP Values (Model-Agnostic, More Reliable)

SHAP (SHapley Additive exPlanations) measures each feature's contribution to each prediction — more reliable than built-in importance for correlated features.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Detailed feature impact
shap.summary_plot(shap_values, X_test)

# Mean absolute SHAP per feature
mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)
top_features = shap_importance.head(20).index.tolist()
```

### L1 Regularization (Lasso)

L1 penalty drives weak feature coefficients to exactly zero — automatic feature selection built into the model.

```python
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import SelectFromModel

# Lasso for regression
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X_train)

# L1 Logistic Regression for classification
lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1)
lr.fit(X_train, y_train)
selected = X.columns[lr.coef_[0] != 0]
```

---

## 7. Decision Framework

```
Start with all features
   ↓
Remove constant / near-constant  →  VarianceThreshold
   ↓
Remove duplicates / highly correlated  →  corr() > 0.95
   ↓
Quick filter  →  Mutual Information / F-test
   ↓
Model-based  →  Feature Importance / SHAP
   ↓
Validate  →  cross-val score with selected features vs all features
```

---

## 8. Summary Table

| Method | Speed | Considers Target | Handles Interactions | Best For |
|---|---|---|---|---|
| Variance Threshold | Fastest | No | No | Quick cleanup |
| Correlation Filter | Fast | No | No | Remove redundancy |
| Chi² / F-test | Fast | Yes | No | Linear relationships |
| Mutual Information | Medium | Yes | Yes | Any relationship type |
| RFE | Slow | Yes | Yes | Small-medium datasets |
| Tree Importance | Medium | Yes | Partial | Tree models |
| SHAP | Slow | Yes | Yes | Any model, most reliable |
| L1 / Lasso | Fast | Yes | No | Linear models |