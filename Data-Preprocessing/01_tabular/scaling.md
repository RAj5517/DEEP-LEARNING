# Scaling and Normalization

Raw numeric features can span wildly different ranges: income in the tens of thousands, age between 0–120, pixel values from 0–255. Many machine learning algorithms are sensitive to the scale of features. This guide covers every major scaling technique, the mathematics behind them, and exactly when to apply each.

---

## 1. Why Scaling Matters

### 1.1 Gradient-Based Optimization
Algorithms like Stochastic Gradient Descent (SGD), Adam, and RMSProp follow the gradient of the loss surface. When features have very different scales, the loss surface becomes elongated (an elliptical bowl instead of a sphere), causing:

- Slow convergence because gradients oscillate in large-scale dimensions.
- The need for a very small learning rate to avoid overshooting.
- Poor conditioning of the optimization problem.

With standardized features, the loss surface is more spherical and gradient descent converges much faster.

### 1.2 Distance-Based Algorithms
Algorithms that compute distances between points — KNN, K-Means, SVM with RBF kernel, PCA — are dominated by features with large scales unless you normalize first.

```
Distance(A, B) = sqrt((income_A - income_B)^2 + (age_A - age_B)^2)
# income: 50000–120000 → dominates the distance entirely
# age: 20–80 → nearly invisible in the calculation
```

### 1.3 Regularization
L1 and L2 regularization penalize the magnitude of coefficients. If features are on different scales, the penalty is applied unfairly across features.

### 1.4 When Scaling is NOT Needed
- **Tree-based models** (Decision Trees, Random Forests, XGBoost, LightGBM) — splits are based on thresholds, not distances or gradients. Scaling has no effect.
- **Naive Bayes** — treats features independently.
- **Rule-based systems**.

---

## 2. Technique 1 — Standardization (Z-Score Normalization)

Transforms each feature to have **zero mean and unit variance** (standard deviation = 1).

### Formula

```
x' = (x - μ) / σ

where:
  μ = mean of the feature
  σ = standard deviation of the feature
```

### Result
- Mean of transformed feature = 0
- Standard deviation of transformed feature = 1
- Values typically fall in the range [-3, 3] for normally distributed data.
- No hard bounds — outliers can produce values far outside [-3, 3].

### Implementation

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()

# CRITICAL: fit on training data only, transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # uses μ and σ from training set

# Manual implementation
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train_scaled = (X_train - mu) / sigma
X_test_scaled = (X_test - mu) / sigma

# Inspect learned parameters
print("Means:", scaler.mean_)
print("Stds:", scaler.scale_)
```

### When to Use
- **Neural networks** — the gold standard preprocessing step; works with SGD, Adam, RMSProp.
- **Linear models** — Logistic Regression, Linear SVM, Ridge, Lasso.
- **PCA and dimensionality reduction** — ensures all features contribute equally to variance.
- **KNN and K-Means** — equalizes distance contributions.
- When the feature distribution is approximately Gaussian.

### Pros
- Preserves the shape of the original distribution.
- Handles outliers better than Min-Max scaling (doesn't collapse normal values into a tiny range).
- Well-suited for gradient-based optimization.

### Cons
- Does not bound values to a specific range — inputs to sigmoid or tanh activations can still be large if outliers exist.
- Sensitive to extreme outliers in the mean and std calculation (use Robust Scaling if outliers are severe).

---

## 3. Technique 2 — Min-Max Scaling (Normalization)

Scales all values to a fixed range, typically **[0, 1]**.

### Formula

```
x' = (x - min(x)) / (max(x) - min(x))

To scale to an arbitrary range [a, b]:
x' = a + (x - min(x)) * (b - a) / (max(x) - min(x))
```

### Result
- Minimum value maps to 0 (or `a`)
- Maximum value maps to 1 (or `b`)
- All values are bounded within the target range.

### Implementation

```python
from sklearn.preprocessing import MinMaxScaler

# Default: [0, 1]
scaler = MinMaxScaler()

# Custom range, e.g., [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inspect
print("Data min:", scaler.data_min_)
print("Data max:", scaler.data_max_)
print("Scale:", scaler.scale_)
```

### When to Use
- **Image data** — pixel values are naturally bounded [0, 255], scaling to [0, 1] is standard.
- **Neural networks with sigmoid or tanh activations** — these saturate outside [0,1] and [-1,1] respectively; bounded inputs prevent saturation.
- **Autoencoders** with sigmoid output layer — output must match input range.
- When your algorithm explicitly requires bounded input (e.g., some SVMs, neural nets with constrained layers).

### Pros
- Intuitive — values are directly interpretable as fractions of the range.
- Guaranteed bounded output.

### Cons
- **Extremely sensitive to outliers.** A single outlier at the extreme compresses all other values into a tiny sub-range:

```python
# Feature: [1, 2, 3, 4, 5, 10000]
# After MinMax: [0, 0.0001, 0.0002, 0.0003, 0.0004, 1.0]
# All normal values are squashed near 0!
```
- At inference time, a value outside [min, max] seen during training produces a value outside [0, 1], breaking the assumption.

---

## 4. Technique 3 — Robust Scaling

Uses the **median and interquartile range (IQR)** instead of mean and standard deviation, making it resistant to outliers.

### Formula

```
x' = (x - median(x)) / IQR(x)

where IQR = Q3 - Q1 = 75th percentile - 25th percentile
```

### Result
- Median maps to 0.
- Values are scaled relative to the spread of the middle 50% of data.
- Outliers are not eliminated, but they don't distort the scaling of normal values.

### Implementation

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(
    quantile_range=(25.0, 75.0),  # default IQR (25th to 75th percentile)
    with_centering=True,           # subtract the median
    with_scaling=True              # divide by IQR
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom quantile range (e.g., 10th to 90th percentile)
scaler = RobustScaler(quantile_range=(10.0, 90.0))
```

### When to Use
- **When outliers are present** and you cannot or don't want to remove them.
- Financial data (income, prices, transaction amounts) — these commonly have heavy tails.
- Medical data (lab values, biomarkers) — clinical outliers are real and meaningful.
- Any domain where extreme values carry important information.

### Pros
- Not distorted by extreme values.
- More statistically robust than StandardScaler for skewed distributions.

### Cons
- Output is not guaranteed to be bounded.
- Slightly more complex to explain than mean/std standardization.
- If data has no true outliers, its advantage over StandardScaler is minimal.

---

## 5. Technique 4 — MaxAbs Scaling

Scales each feature by dividing by the **maximum absolute value**, mapping to the range [-1, 1] without shifting the data (preserves sparsity).

### Formula

```
x' = x / max(|x|)
```

### Implementation

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### When to Use
- **Sparse data** (e.g., TF-IDF matrices, sparse feature vectors from text).
- When centering (subtracting mean) would destroy sparsity (zeros would become non-zero).

---

## 6. Technique 5 — L1 / L2 Normalization (Row Normalization)

Normalizes **each sample (row)** to have unit norm, rather than scaling each feature (column).

### Formula

```
L2 normalization: x' = x / ||x||₂     (Euclidean norm)
L1 normalization: x' = x / ||x||₁     (Manhattan norm)
```

### Implementation

```python
from sklearn.preprocessing import Normalizer

# L2 norm (default) — makes each sample a unit vector
normalizer = Normalizer(norm="l2")
X_normalized = normalizer.transform(X)

# L1 norm — each sample sums to 1 (useful for proportions)
normalizer = Normalizer(norm="l1")
```

### When to Use
- **Text classification with TF-IDF** — document length varies; normalizing rows makes documents comparable.
- **Cosine similarity** — effectively computes the angle between vectors, making row magnitude irrelevant.
- When the direction of a feature vector matters more than its magnitude.

### Caution
This is very different from column-wise scaling. Use it only when the relative proportions within each sample are what matter.

---

## 7. Technique 6 — Log / Power Transformations

These are not scalers in the strict sense, but **feature transformations** that change the distribution's shape before scaling. They handle heavily skewed data.

### Log Transformation

```python
import numpy as np

# Log transform (for positive data)
df["income_log"] = np.log1p(df["income"])  # log1p = log(1 + x), handles x=0

# Then standardize
from sklearn.preprocessing import StandardScaler
df["income_scaled"] = StandardScaler().fit_transform(df[["income_log"]])
```

### Box-Cox Transformation

Finds the optimal power transformation λ to make the distribution as Gaussian as possible. Requires all values > 0.

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="box-cox")   # requires positive values
pt = PowerTransformer(method="yeo-johnson")  # works with negative values too
X_transformed = pt.fit_transform(X)
```

### When to Use
- Features with **highly right-skewed distributions** (income, population, price).
- When a feature spans multiple orders of magnitude.
- Before applying StandardScaler, if the distribution is far from Gaussian.

---

## 8. The Data Leakage Problem — The Most Critical Rule

**Always fit scalers on training data only. Never on the full dataset including test data.**

```python
# ❌ WRONG — data leakage: test statistics influence training scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # using all data including test set

# ✅ CORRECT — proper pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
# The pipeline automatically fits scaler on X_train and transforms X_test
```

Fitting the scaler on test data leaks test statistics (mean, std, min, max) into the training process, leading to inflated performance metrics that don't reflect real-world performance.

---

## 9. Scaling in Neural Networks — Layer Normalization Techniques

Beyond preprocessing, modern deep learning uses **internal normalization layers** to keep activations in good ranges throughout training.

### 9.1 Batch Normalization

Normalizes activations within a mini-batch across the batch dimension. Applied after a linear layer and before (or after) the activation function.

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),   # normalizes across the batch dimension
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU()
)
```

Benefits: faster training, higher learning rates, acts as regularization, reduces sensitivity to initialization.

### 9.2 Layer Normalization

Normalizes across the feature dimension for a single sample. Preferred for RNNs and Transformers because it doesn't depend on batch size.

```python
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.LayerNorm(64),    # normalizes across features for each sample independently
    nn.ReLU()
)
```

### 9.3 Group Normalization

Divides features into groups and normalizes within each group. Useful for small batch sizes where Batch Norm is noisy.

```python
nn.GroupNorm(num_groups=8, num_channels=64)
```

---

## 10. Inverse Transforming Predictions

When the **target variable (y)** is also scaled (e.g., in regression), you must inverse-transform predictions back to the original scale before evaluating or reporting.

```python
from sklearn.preprocessing import StandardScaler

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

# Train model on scaled targets
model.fit(X_train_scaled, y_train_scaled)

# Predict and inverse transform
y_pred_scaled = model.predict(X_test_scaled)
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Evaluate in original scale
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: ${mae:,.2f}")
```

---

## 11. Decision Framework

```
Does your model use gradient descent or distance calculations?
├── NO (Trees, Rule-based) → Scaling not required
└── YES
    ├── Is the data sparse (text/TF-IDF)?
    │   └── YES → MaxAbs Scaling
    ├── Does each sample need a unit norm (cosine similarity)?
    │   └── YES → L2 Normalizer (row-wise)
    ├── Are there significant outliers in the data?
    │   └── YES → Robust Scaling (median + IQR)
    ├── Is input range bounded (images, signals)?
    │   └── YES → Min-Max Scaling [0,1] or [-1,1]
    ├── Is the distribution heavily skewed?
    │   └── YES → Log / Box-Cox transform → then StandardScaler
    └── Default case → StandardScaler (Z-score)
```

---

## 12. Summary Table

| Technique | Formula | Output Range | Outlier Sensitive | Best For |
|---|---|---|---|---|
| Z-Score (Standard) | `(x - μ) / σ` | Unbounded, ~[-3,3] | Moderate | Neural nets, Linear models, PCA |
| Min-Max | `(x - min) / (max - min)` | [0, 1] or [a, b] | High | Images, bounded activations |
| Robust | `(x - median) / IQR` | Unbounded | Low | Outlier-heavy data |
| MaxAbs | `x / max(|x|)` | [-1, 1] | Moderate | Sparse data |
| L2 Normalizer | `x / ||x||₂` | Unit norm per row | N/A | Text, cosine similarity |
| Log Transform | `log(1 + x)` | Varies | Reduces skew | Skewed distributions |
| Box-Cox / Yeo-Johnson | Optimal power λ | Varies | Reduces skew | Making distributions Gaussian |

---

## 13. Complete End-to-End Example

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

numeric_features = ["age", "income", "score"]
image_features = ["pixel_brightness"]
robust_features = ["transaction_amount"]
categorical_features = ["city", "education"]

preprocessor = ColumnTransformer(transformers=[
    ("standard", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("minmax", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ]), image_features),
    ("robust", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ]), robust_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features),
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", GradientBoostingClassifier(n_estimators=100))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")
```