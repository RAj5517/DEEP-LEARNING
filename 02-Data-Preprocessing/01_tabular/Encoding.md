# Encoding Categorical Variables

Neural networks and most machine learning algorithms operate on numbers. Categorical variables — text labels, classes, or nominal groups — must be converted into a numeric representation before training. Choosing the right encoding can dramatically affect model performance and training efficiency.

---

## 1. Types of Categorical Variables

Understanding the nature of your categories shapes which encoding to use.

### 1.1 Nominal (Unordered)
Categories with **no inherent order**. The numeric representation must not impose a false ordering.

Examples: `color = {red, green, blue}`, `country = {India, USA, Germany}`

### 1.2 Ordinal (Ordered)
Categories with a **meaningful rank or order**, but the distances between ranks may not be equal.

Examples: `education = {High School < Bachelor's < Master's < PhD}`, `rating = {Low < Medium < High}`

### 1.3 High-Cardinality Categorical
A column with a **very large number of unique values** (hundreds to millions).

Examples: `zip_code`, `user_id`, `product_SKU`, `city`

### 1.4 Binary Categorical
A column with exactly two possible values.

Examples: `gender = {Male, Female}`, `has_subscription = {Yes, No}`

---

## 2. Encoding Strategy 1 — Label Encoding (Ordinal Encoding)

Assigns an integer to each category. Order is imposed, which is appropriate for **ordinal** categories.

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# LabelEncoder: for a single column
le = LabelEncoder()
df["education_encoded"] = le.fit_transform(df["education"])

# OrdinalEncoder: for multiple columns, with custom ordering
categories = [["High School", "Bachelor's", "Master's", "PhD"]]
enc = OrdinalEncoder(categories=categories)
df[["education_encoded"]] = enc.fit_transform(df[["education"]])
```

### When to Use
- The variable is **ordinal** and the order is meaningful.
- Tree-based models (Random Forest, XGBoost, LightGBM) — they naturally split on thresholds and don't assume linearity, so arbitrary integer encoding often works fine even for nominal categories.

### When NOT to Use
- **Linear models or neural networks** with nominal categories — the integer encoding implies a false ordering (e.g., `red=0, green=1, blue=2` implies green is between red and blue).

---

## 3. Encoding Strategy 2 — One-Hot Encoding (OHE)

Creates a new binary column for each unique category. The original column is replaced with k binary columns, where k is the number of unique categories. Exactly one column is `1` for each row.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# pandas approach
df_encoded = pd.get_dummies(df, columns=["color"], drop_first=False)

# sklearn approach (pipeline-friendly)
enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = enc.fit_transform(df[["color"]])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(["color"]))
```

### drop="first" — Avoiding the Dummy Variable Trap

For linear models, one column is perfectly predictable from the others, causing multicollinearity. Dropping one category fixes this. For tree models and neural networks, dropping is optional.

```python
# 3 categories → 2 columns (drop="first")
# red=0, green=0 → implies blue
# red=1, green=0 → red
# red=0, green=1 → green
```

### When to Use
- **Small cardinality** (< ~15–20 unique values).
- **Linear models** (Logistic Regression, Linear SVM, Ridge).
- **Neural networks** with small categorical features.
- When you need a simple, interpretable representation.

### Cons
- Creates **very wide, sparse matrices** for high-cardinality columns.
- Loses all information about relationships between categories.
- New categories at inference time must be handled with `handle_unknown="ignore"`.

---

## 4. Encoding Strategy 3 — Binary Encoding

Converts each integer-encoded category into its binary representation, using only `log2(k)` columns instead of `k` columns. Efficient for medium-cardinality variables.

```python
import category_encoders as ce

enc = ce.BinaryEncoder(cols=["city"])
df_encoded = enc.fit_transform(df)
```

A column with 100 unique cities produces only 7 columns instead of 100.

### When to Use
- **Medium-cardinality** categories (20–200 unique values) where OHE creates too many columns.
- Tree-based models where a compact representation is preferred.

---

## 5. Encoding Strategy 4 — Target Encoding (Mean Encoding)

Replaces each category with the **mean of the target variable** for that category. The model directly receives a numeric signal derived from label-category correlation.

```python
import category_encoders as ce

enc = ce.TargetEncoder(cols=["city"], smoothing=10)
enc.fit(X_train, y_train)
X_train_encoded = enc.transform(X_train)
X_test_encoded = enc.transform(X_test)
```

### Smoothing
Without smoothing, rare categories get noisy estimates. Smoothing blends the category mean toward the global mean:

```
encoded_value = (n * category_mean + k * global_mean) / (n + k)
```

where `n` is the count of that category and `k` is the smoothing factor.

### Critical Warning: Target Leakage
If target encoding is computed using the same rows you train on, the model sees the label information directly, causing overfitting. Always use:

```python
# Option 1: Fit encoder on training set only
enc.fit(X_train, y_train)
X_train_encoded = enc.transform(X_train)  # slight leakage still exists on train set

# Option 2: K-Fold target encoding (better)
enc = ce.TargetEncoder(cols=["city"], smoothing=10)
# Use cross_val_predict-style transformation to avoid leakage
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

### When to Use
- **High-cardinality** columns (zip code, product ID, city).
- Gradient boosting models (XGBoost, LightGBM, CatBoost).
- When OHE would create thousands of sparse columns.

### Cons
- Leakage risk on training set.
- Can overfit to the target distribution.
- Loses information about low-frequency categories.

---

## 6. Encoding Strategy 5 — Frequency / Count Encoding

Replaces each category with how many times it appears in the dataset (or its proportion). Simple and effective for high-cardinality variables.

```python
# Frequency encoding
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)

# Count encoding
count = df["city"].value_counts()
df["city_count"] = df["city"].map(count)
```

### When to Use
- When the **frequency of a category** is itself predictive.
- Tree-based models that benefit from compact numeric signals.
- Fast, zero-leakage encoding.

### Cons
- Two categories with the same frequency get the same code (collision).
- Doesn't capture the relationship with the target.

---

## 7. Encoding Strategy 6 — Embeddings (Neural Network Approach)

For large-cardinality categorical variables, embeddings learn a **dense vector representation** during training. Unlike OHE, the model learns what each category means in a lower-dimensional space.

### 7.1 How Embeddings Work

Each category is mapped to an integer index, and an `Embedding` layer converts that index into a trainable vector of size `d` (embedding dimension).

```
Category: "New York" → Index: 42 → Embedding Vector: [0.23, -0.11, 0.87, ...]
```

### 7.2 Implementation in PyTorch

```python
import torch
import torch.nn as nn

num_cities = 500      # number of unique cities
embedding_dim = 16    # embedding size (rule of thumb: min(50, (n_cat // 2) + 1))

city_embedding = nn.Embedding(num_embeddings=num_cities + 1, embedding_dim=embedding_dim, padding_idx=0)

# Forward pass
city_indices = torch.LongTensor([42, 7, 255])   # batch of city indices
city_vectors = city_embedding(city_indices)       # shape: (3, 16)
```

### 7.3 Implementation in TensorFlow / Keras

```python
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Concatenate
from tensorflow.keras.models import Model

city_input = Input(shape=(1,), name="city")
city_emb = Embedding(input_dim=501, output_dim=16, name="city_embedding")(city_input)
city_flat = Flatten()(city_emb)

# Combine with other features
other_input = Input(shape=(10,), name="other_features")
concat = Concatenate()([city_flat, other_input])
output = Dense(1, activation="sigmoid")(concat)

model = Model(inputs=[city_input, other_input], outputs=output)
```

### 7.4 Choosing Embedding Dimension

A common rule of thumb: `embedding_dim = min(50, (num_unique_categories // 2) + 1)`

| Unique Categories | Suggested Embedding Dim |
|---|---|
| 5–10 | 3–5 |
| 50–100 | 25–50 |
| 500–1000 | 50 |
| 10,000+ | 50–100 |

### 7.5 Pretrained / Transfer Embeddings

For text and language features, use pretrained embeddings that encode semantic similarity:

```python
# Word2Vec, FastText, GloVe for text
# Use Hugging Face for contextual embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["New York", "Los Angeles", "Chicago"])
```

### When to Use Embeddings
- **High-cardinality** categories (hundreds to millions of unique values).
- Neural network models where end-to-end learning is possible.
- When semantic or relational structure between categories matters.
- Recommended when OHE would produce thousands of sparse columns.

### Cons
- Requires enough training data for each category to learn a meaningful vector.
- Not directly interpretable.
- Overkill for small cardinality variables.

---

## 8. Encoding Strategy 7 — Hashing Encoding (Feature Hashing)

Uses a hash function to map categories to a fixed number of buckets. Handles high-cardinality and unknown categories gracefully without needing a vocabulary.

```python
from sklearn.feature_extraction import FeatureHasher

hasher = FeatureHasher(n_features=64, input_type="string")
hashed = hasher.transform(df["city"].apply(lambda x: {x: 1}))
```

### When to Use
- **Very high-cardinality** with large or unknown vocabularies.
- Online learning where the full category list is not known upfront.
- Memory-constrained environments.

### Cons
- Hash collisions (two categories map to the same bucket).
- Loss of interpretability.
- Cannot reverse the encoding.

---

## 9. Handling Unknown Categories at Inference

A critical production concern: what if a category at inference time was never seen during training?

```python
# OneHotEncoder: ignore unknowns
enc = OneHotEncoder(handle_unknown="ignore")

# OrdinalEncoder: assign a special value
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# Embeddings: reserve index 0 as an "unknown" token
city_embedding = nn.Embedding(num_embeddings=num_cities + 1, embedding_dim=16, padding_idx=0)
# At inference: map unseen categories → index 0
```

---

## 10. Decision Framework

```
What is the cardinality?
├── Binary (2 values) → Label Encoding (0/1)
├── Low (< 15 values)
│   ├── Ordinal variable? → Ordinal Encoding
│   └── Nominal + Linear/NN model? → One-Hot Encoding
├── Medium (15–200 values)
│   ├── Tree model? → Label / Binary Encoding
│   └── Linear model? → Binary Encoding or OHE
└── High (200+ values)
    ├── Tree model? → Target Encoding or Frequency Encoding
    └── Neural Network? → Embeddings
```

---

## 11. Summary Table

| Encoding | Cardinality | Model Type | Pros | Cons |
|---|---|---|---|---|
| Label / Ordinal | Low–Medium | Trees | Simple, compact | False ordering for nominal |
| One-Hot | Low (< 15) | Any | No false ordering | Sparse, high-dimensional |
| Binary | Medium | Trees | Compact | Not intuitive |
| Target | High | Trees | Strong signal | Leakage risk |
| Frequency / Count | High | Trees | Simple, no leakage | Collisions |
| Embeddings | High | Neural Nets | Learns relationships | Needs data, not interpretable |
| Hashing | Very High | Any | Memory efficient | Collisions, not reversible |