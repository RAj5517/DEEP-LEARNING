# Time Series Transformers

> "Stock prices. Weather. Sensor readings. Patient vitals. The world runs on sequences through time — and transformers were made for sequences."

---

## Why Time Series is Different

Time series data has unique properties that make it both natural and challenging for transformers:

```
Time Series Challenges
│
├── Temporal ordering matters        past causes future, not vice versa
├── Multiple scales                  hourly patterns + daily + seasonal + trend
├── Multivariate dependencies        temperature affects humidity affects wind speed
├── Long-range dependencies          seasonal patterns repeat every 365 days
└── Non-stationarity                 distribution shifts over time
```

Standard transformers were designed for NLP. Applying them naively to time series often fails — the positional encoding doesn't capture temporal periodicity well, and patch-level patterns matter differently.

---

## How Time Steps Become Tokens

```
Time Series → Transformer Input
│
├── Point-wise               each timestep = one token (simple but loses local patterns)
│
├── Patch-based (PatchTST)   group P timesteps into one patch-token
│     └── Better local feature extraction + longer effective range
│
└── Channel mixing           each variable (temperature, humidity) = separate sequence
      └── cross-variable attention to learn dependencies
```

---

## Main Models

### Temporal Fusion Transformer — TFT (2021) — Google Cloud
The most practically used time series transformer in production. Designed specifically for **multi-horizon forecasting** with complex real-world data.

```
TFT Architecture
│
├── Input Processing
│     ├── Static covariates        entity-level features (store ID, location)
│     ├── Known future inputs      holidays, planned events
│     └── Observed past inputs     historical sales, weather
│
├── Variable Selection Networks    learn which inputs matter (per timestep)
├── LSTM Encoder                   capture local temporal patterns first
├── Self-Attention                 capture long-range dependencies
│
└── Output
      └── Quantile forecasts       predict 10th, 50th, 90th percentile
                                   (know uncertainty, not just point estimate)
```

Key differentiator: **interpretability**. TFT tells you which variables it used and which time steps mattered most for each prediction — critical for business decisions.
Paper: https://arxiv.org/abs/1912.09363

---

### PatchTST (2023) — Yuqi Nie et al.
Applied Vision Transformer (ViT) ideas to time series. Instead of treating each timestep as a token, it creates **patches** of L consecutive timesteps.

```
PatchTST Key Ideas
│
├── Patching               group 16 timesteps → one patch-token
│     └── Reduces sequence length → faster attention · better local patterns
│
├── Channel-Independent    each variable processed separately
│     └── No cross-variable interference during pre-training
│
└── Masked Pre-training    mask random patches → predict them
      └── Same as MAE (Masked Autoencoders) for images
```

Outperformed much larger models on long-term forecasting benchmarks with this simple approach.
Paper: https://arxiv.org/abs/2211.14730

---

### TimesFM (2024) — Google
**Foundation model for time series** — the GPT moment for time series forecasting. Pre-trained on a large corpus of time series data (100B+ time points from Google Trends, Wikipedia, Finance, etc.).

Zero-shot forecasting: give it any new time series it has never seen, and it forecasts forward without fine-tuning. Like GPT for text, but for temporal data.

Paper: https://arxiv.org/abs/2310.10688

---

## Other Notable Models

| Model | By | Note |
|-------|----|------|
| Autoformer | Tsinghua | Decomposition + auto-correlation for long-range forecasting |
| Informer | Beihang | Efficient attention for very long sequences (ProbSparse attention) |
| TimesNet | Shanghai AI Lab | Convert 1D time series to 2D then apply 2D convolutions |
| Moirai | Salesforce | Universal forecasting transformer — handles any frequency |
| Chronos | Amazon | LLM-based time series forecasting via tokenizing numerical values |

---

## Use Cases

```
Time Series Transformer Use Cases
│
├── Demand Forecasting           retail sales · energy consumption · supply chain
├── Financial Forecasting        stock prices · volatility · risk modeling
├── Weather & Climate            temperature · precipitation · extreme event prediction
├── Anomaly Detection            equipment failure · fraud · network intrusion
├── Healthcare                   patient vitals · ICU monitoring · EHR prediction
├── Traffic Forecasting          road congestion · public transport load
└── IoT / Industrial             sensor readings · predictive maintenance
```

---

## TFT vs PatchTST vs TimesFM

```
Model Comparison
│
├── TFT
│     ├── Best for: complex real-world forecasting with mixed input types
│     ├── Strength: interpretable · handles static + dynamic + known future inputs
│     └── Use when: you need to explain predictions to stakeholders
│
├── PatchTST
│     ├── Best for: long-horizon univariate / multivariate forecasting
│     ├── Strength: simple · efficient · strong benchmark performance
│     └── Use when: you have clean numerical time series data
│
└── TimesFM
      ├── Best for: new domains where you have little training data
      ├── Strength: zero-shot forecasting — no fine-tuning needed
      └── Use when: you need quick results on a new dataset
```

---

## Key Papers

| Paper | Link |
|-------|------|
| Temporal Fusion Transformer (2021) | https://arxiv.org/abs/1912.09363 |
| Informer (2021) | https://arxiv.org/abs/2012.07436 |
| PatchTST (2023) | https://arxiv.org/abs/2211.14730 |
| TimesFM (2024) | https://arxiv.org/abs/2310.10688 |