# Handling Missing Values in Machine Learning

Missing values are one of the most common problems in real-world datasets. How you handle them can significantly impact your model's performance. This guide covers every major strategy, when to use each, and their trade-offs.

---

## 1. Understanding Missing Data

Before treating missing values, it's critical to understand **why** they are missing. There are three types of missingness:

### 1.1 Missing Completely at Random (MCAR)
The probability of a value being missing has **no relationship** with any other variable in the dataset — it's purely random (e.g., a sensor randomly drops a reading).

- **Impact:** Least harmful; any removal or imputation method works.
- **Detection:** Run Little's MCAR test or compare distributions of complete vs. incomplete subsets.

### 1.2 Missing at Random (MAR)
The probability of a value being missing **depends on other observed variables**, but not on the missing value itself (e.g., older patients are less likely to fill out an online form).

- **Impact:** Moderate risk; model-based imputation is preferred.
- **Detection:** Check correlations between missingness indicators and other features.

### 1.3 Missing Not at Random (MNAR)
The missingness **depends on the missing value itself** (e.g., high-income individuals refuse to disclose their salary).

- **Impact:** Most dangerous; simple imputation introduces bias.
- **Recommended approach:** Domain knowledge, separate modeling, or keeping a missingness indicator.

---

## 2. Exploratory Analysis of Missing Data

Always begin by profiling your missing data:

```python
import pandas as pd
import missingno as msno

df = pd.read_csv("data.csv")

# Count and percentage of missing values
missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing %": df.isnull().mean() * 100
}).sort_values("Missing %", ascending=False)

print(missing_summary[missing_summary["Missing Count"] > 0])

# Visual matrix of missing patterns
msno.matrix(df)
msno.heatmap(df)  # shows correlation between missingness of columns
```

---

## 3. Strategy 1 — Remove Rows (Listwise Deletion)

Drop any row that contains at least one missing value.

```python
df_clean = df.dropna()

# Or drop only if specific columns are missing
df_clean = df.dropna(subset=["age", "income"])
```

### When to Use
- Data is **MCAR** and the fraction missing is **small** (< 5%).
- Losing rows won't create statistical bias.

### Pros
- Simple and fast.
- No synthetic data is introduced.

### Cons
- Loses potentially useful data.
- Can introduce bias if data is MAR or MNAR.
- Reduces statistical power.

---

## 4. Strategy 2 — Remove Columns

If a column has an extremely high percentage of missing values (e.g., > 70%), it may carry little useful signal.

```python
threshold = 0.7  # drop if > 70% missing
df_clean = df.loc[:, df.isnull().mean() < threshold]
```

### When to Use
- A feature is sparsely populated and unlikely to contribute signal.
- The feature is redundant with other available features.

### Caution
- Never drop without checking: sometimes a high-missing-rate column is actually very informative (its very missingness is the signal).

---

## 5. Strategy 3 — Mean / Median / Mode Imputation

Replace missing values with a summary statistic calculated from the non-missing values of the same column.

```python
from sklearn.impute import SimpleImputer
import numpy as np

# Mean imputation (continuous features)
imputer = SimpleImputer(strategy="mean")

# Median imputation (robust to outliers)
imputer = SimpleImputer(strategy="median")

# Mode imputation (categorical or discrete features)
imputer = SimpleImputer(strategy="most_frequent")

# Constant imputation
imputer = SimpleImputer(strategy="constant", fill_value=0)

df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

### Mean vs. Median
| | Mean | Median |
|---|---|---|
| Sensitive to outliers | Yes | No |
| Best for | Symmetric distributions | Skewed distributions |

### When to Use
- Data is MCAR or MAR with low missingness rate.
- Quick baseline before trying more sophisticated methods.

### Cons
- **Distorts the distribution** — reduces variance artificially.
- **Ignores relationships** between features (a value imputed from the global mean ignores what other columns say about that row).
- Poor for MNAR data.

---

## 6. Strategy 4 — Forward Fill / Backward Fill (Time Series)

In temporal or sequential data, propagate the last known value forward (or backward).

```python
# Forward fill (use last observed value)
df_filled = df.fillna(method="ffill")

# Backward fill (use next observed value)
df_filled = df.fillna(method="bfill")

# Limit how far to fill
df_filled = df.fillna(method="ffill", limit=3)
```

### When to Use
- **Time series** data where consecutive readings are expected to be similar (sensor data, stock prices, IoT streams).

### Cons
- Can propagate stale data over long gaps.
- Inappropriate for cross-sectional (non-temporal) data.

---

## 7. Strategy 5 — Interpolation (Time Series / Ordered Data)

Estimate missing values by interpolating between known neighboring values.

```python
# Linear interpolation
df_interp = df.interpolate(method="linear")

# Polynomial or spline interpolation for smoother curves
df_interp = df.interpolate(method="spline", order=3)

# Time-aware interpolation (requires DatetimeIndex)
df.index = pd.to_datetime(df.index)
df_interp = df.interpolate(method="time")
```

### When to Use
- Sequential or time-indexed data with gradual trends.
- When a smooth estimate between points makes physical or logical sense.

---

## 8. Strategy 6 — Model-Based Imputation (Iterative / MICE)

Use machine learning models to **predict** missing values from other features in the dataset. This is the most powerful and statistically principled approach.

### 8.1 KNN Imputation

Find the k-nearest neighbors of a row (based on non-missing features) and impute from their values.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights="uniform")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

### 8.2 Iterative Imputation (MICE — Multiple Imputation by Chained Equations)

Treats each feature with missing values as a target variable and trains a regression/classification model on the others. Iterates until convergence.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10,
    random_state=42
)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

### When to Use
- Data is MAR and relationships between features are strong.
- You have enough data that training per-column models is feasible.
- Accuracy matters more than speed.

### Pros
- Respects feature correlations.
- Statistically principled for MAR data.
- Can be more accurate than simple imputation by a wide margin.

### Cons
- Computationally expensive.
- Risk of overfitting the imputed values.
- Requires careful cross-validation to avoid data leakage (always fit on training set, transform both train and test).

---

## 9. Strategy 7 — Missing-Category / Indicator Embedding

Instead of removing or filling missing values, **explicitly encode missingness as information**.

### 9.1 Adding a Binary Missingness Indicator

```python
import numpy as np

for col in df.columns:
    if df[col].isnull().any():
        df[f"{col}_was_missing"] = df[col].isnull().astype(int)

# Then impute the original column
df.fillna(df.median(), inplace=True)
```

### 9.2 Missing-Category for Categoricals

Add a new category called `"Unknown"` or `"Missing"` instead of dropping or imputing.

```python
df["education"] = df["education"].fillna("Unknown")
```

### 9.3 Why This Matters for Neural Networks

In embedding-based models (e.g., entity embeddings for categorical variables), a `"Missing"` token can be learned as a specific embedding vector — the model learns what missingness signals about a row without any explicit imputation.

```python
# Example: encode category + missing token
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```

### When to Use
- Data is **MNAR** (the fact of missingness carries predictive signal).
- Tree-based models or neural networks that can learn from an extra feature.
- When you want maximum transparency about what the model has learned.

---

## 10. Strategy 8 — Deep Learning / Generative Imputation

For high-dimensional or complex data (images, text, tabular), generative models can impute plausible values.

- **Autoencoders:** Train an autoencoder on complete data, then use the latent representation to reconstruct missing features.
- **GANs (GAIN):** Generative Adversarial Imputation Networks — a GAN specifically designed for tabular missing data imputation.
- **Diffusion Models:** State-of-the-art for image inpainting (a form of missingness in image data).

These are advanced and typically reserved for cases where simpler methods fail or data has complex structure.

---

## 11. Practical Decision Framework

```
Is the missingness rate < 5%?
├── YES → Listwise deletion (MCAR) or Mean/Median (MAR)
└── NO
    ├── Is the data temporal/sequential?
    │   └── YES → Forward fill, Backfill, or Interpolation
    └── NO
        ├── Are features strongly correlated?
        │   └── YES → KNN or MICE (Iterative Imputation)
        └── Is missingness itself informative (MNAR)?
            └── YES → Missingness Indicator + any imputation
```

---

## 12. Critical Best Practices

**Fit on training data only.** Always call `imputer.fit()` on the training set and use `transform()` on both train and test. Fitting on the test set leaks information and inflates performance metrics.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingClassifier())
])

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

**Document your choices.** Missing value handling is a modeling decision — record which strategy was used, why, and on which columns. This is essential for reproducibility and debugging.

**Validate imputation quality.** Artificially mask known values and measure how accurately your imputation strategy recovers them:

```python
# Create artificial missingness in a complete subset
X_complete = df.dropna()
mask = np.random.rand(*X_complete.shape) < 0.1  # mask 10% of values
X_masked = X_complete.copy().mask(mask)

imputer.fit(X_masked)
X_recovered = imputer.transform(X_masked)

# Compare recovered vs. original
rmse = np.sqrt(np.mean((X_complete.values[mask] - X_recovered[mask])**2))
print(f"Imputation RMSE: {rmse:.4f}")
```

---

## Summary Table

| Strategy | Data Type | Missingness Type | Complexity | Risk |
|---|---|---|---|---|
| Remove Rows | Any | MCAR only | Low | Bias if not MCAR |
| Remove Columns | Any | Very high rate | Low | Loses features |
| Mean/Median/Mode | Numeric / Categorical | MCAR / MAR | Low | Distorts distribution |
| Forward/Backward Fill | Time Series | MCAR / MAR | Low | Stale data propagation |
| Interpolation | Sequential | MCAR / MAR | Low-Medium | Poor for large gaps |
| KNN Imputation | Numeric | MAR | Medium | Sensitive to scale |
| MICE / Iterative | Numeric | MAR | High | Data leakage risk |
| Missingness Indicator | Any | MNAR | Low | Adds dimensionality |
| Generative (GAN/AE) | Complex / Image | Any | Very High | Training instability |