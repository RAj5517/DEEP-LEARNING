# Feature Engineering

Feature engineering is the process of using domain knowledge to create new input features from raw data. It is often the single highest-leverage action you can take to improve model performance — more impactful than tuning architecture or hyperparameters.

> A well-engineered feature can make a simple linear model outperform a complex neural network trained on raw data.

---

## 1. Why Feature Engineering?

Raw data rarely expresses the signal a model needs directly. Feature engineering bridges that gap:

- Raw: `timestamp = 1706745600`
- Engineered: `hour=8, day_of_week=Monday, is_rush_hour=True, month=January`

The engineered features make patterns the model can learn trivially — without needing to discover the periodicity of time from a raw Unix timestamp.

---

## 2. Numerical Feature Transformations

### Polynomial Features
Capture non-linear relationships between features without a non-linear model.

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Creates x, x², x³, x·y, x²·y, y², etc.
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Raw features: [x, y]  →  [x, y, x², x·y, y²]
```

### Binning / Discretization
Convert continuous values into categorical buckets — reduces sensitivity to outliers and captures non-linear thresholds.

```python
import pandas as pd

# Equal-width bins
df["age_bin"] = pd.cut(df["age"], bins=5, labels=["teen","young","mid","senior","elder"])

# Equal-frequency bins (quantile-based)
df["income_bin"] = pd.qcut(df["income"], q=4, labels=["Q1","Q2","Q3","Q4"])

# Custom domain-based bins
df["bmi_category"] = pd.cut(df["bmi"],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=["underweight","normal","overweight","obese"]
)
```

### Log / Power Transforms
Correct skewed distributions before modeling (covered in depth in tabular normalization).

```python
df["log_income"] = np.log1p(df["income"])
df["sqrt_distance"] = np.sqrt(df["distance"])
```

### Ratios and Interactions
Explicitly encode relationships between features the model might not discover.

```python
# Financial features
df["debt_to_income"]   = df["debt"] / (df["income"] + 1e-8)
df["profit_margin"]    = df["profit"] / (df["revenue"] + 1e-8)

# E-commerce features
df["cart_conversion"]  = df["purchases"] / (df["cart_adds"] + 1e-8)
df["revenue_per_user"] = df["revenue"] / (df["users"] + 1e-8)

# Interaction terms
df["area"] = df["length"] * df["width"]
df["speed"] = df["distance"] / (df["time"] + 1e-8)
```

---

## 3. Datetime Features

Time contains rich, periodic signals that must be unpacked.

```python
df["datetime"] = pd.to_datetime(df["timestamp"])

# Extract components
df["hour"]        = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek    # 0=Monday
df["day_of_month"]= df["datetime"].dt.day
df["month"]       = df["datetime"].dt.month
df["quarter"]     = df["datetime"].dt.quarter
df["year"]        = df["datetime"].dt.year
df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)

# Cyclic encoding — preserves that hour 23 is close to hour 0
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Time since a reference event
df["days_since_signup"] = (df["datetime"] - df["signup_date"]).dt.days
```

---

## 4. Text-Derived Features

```python
df["text_length"]    = df["review"].str.len()
df["word_count"]     = df["review"].str.split().str.len()
df["avg_word_len"]   = df["text_length"] / (df["word_count"] + 1e-8)
df["exclamation_ct"] = df["review"].str.count("!")
df["question_ct"]    = df["review"].str.count(r"\?")
df["capital_ratio"]  = df["review"].apply(lambda x: sum(c.isupper() for c in x) / (len(x) + 1))
df["has_url"]        = df["review"].str.contains(r"http").astype(int)
```

---

## 5. Aggregation Features (Group Statistics)

Capture behavior relative to a group — powerful for tabular ML on user/entity data.

```python
# Mean target encoding by group (careful: compute on train only)
group_stats = df.groupby("city")["purchase_amount"].agg(
    city_mean_purchase = "mean",
    city_std_purchase  = "std",
    city_count         = "count"
).reset_index()

df = df.merge(group_stats, on="city", how="left")

# Rolling statistics for time series
df = df.sort_values(["user_id", "date"])
df["rolling_7d_spend"] = df.groupby("user_id")["spend"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
```

---

## 6. Lag Features (Time Series)

Bring past values into the current row — essential for forecasting.

```python
df = df.sort_values("date")

# Lag features
for lag in [1, 7, 14, 30]:
    df[f"sales_lag_{lag}"] = df["sales"].shift(lag)

# Rolling window features
df["sales_rolling_7d_mean"] = df["sales"].rolling(7).mean()
df["sales_rolling_7d_std"]  = df["sales"].rolling(7).std()

# Exponential moving average
df["sales_ema_7"] = df["sales"].ewm(span=7, adjust=False).mean()
```

---

## 7. Domain-Specific Examples

### E-commerce
```python
df["recency"]   = (today - df["last_purchase_date"]).dt.days
df["frequency"] = df["purchase_count"] / df["days_as_customer"]
df["monetary"]  = df["total_spend"] / df["purchase_count"]
# → RFM (Recency, Frequency, Monetary) — classic customer segmentation features
```

### Finance
```python
df["price_change"]    = df["close"] - df["open"]
df["price_range"]     = df["high"] - df["low"]
df["volatility_20d"]  = df["returns"].rolling(20).std()
df["rsi"]             = compute_rsi(df["close"], period=14)  # momentum indicator
```

### Healthcare
```python
df["bmi"]          = df["weight_kg"] / (df["height_m"] ** 2)
df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
df["age_bmi"]      = df["age"] * df["bmi"]   # interaction
```

---

## 8. Automated Feature Engineering

```python
# Featuretools — automated relational feature synthesis
import featuretools as ft

es = ft.EntitySet(id="customers")
es.add_dataframe(dataframe=df, dataframe_name="orders",
                 index="order_id", time_index="order_date")

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="orders",
    max_depth=2
)
```