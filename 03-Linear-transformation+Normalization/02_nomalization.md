# Normalization — BatchNorm · LayerNorm · RMSNorm

---

## Why Normalization Is Needed

During training, as weights update, the distribution of inputs to each layer keeps changing. A layer trained to expect inputs with mean ≈ 0 and std ≈ 1 suddenly receives inputs with mean = 5 and std = 0.2 — it has to constantly re-adapt.

This is called **Internal Covariate Shift** — the distribution of layer inputs shifts internally during training.

**Problems it causes:**
```
→ Training requires very small learning rates (or it diverges)
→ Weight initialization becomes critical
→ Saturating activations (Sigmoid/Tanh) get stuck in flat regions
→ Deep networks train extremely slowly
```

**What normalization does:**
Re-centers and re-scales activations to a stable distribution after the linear transform. This stabilizes training, allows higher learning rates, and acts as a regularizer.

---

## The Shared Mathematical Skeleton

All three normalization types follow the same structure. The only difference is **which values you average over**.

```
1. Compute mean   μ   over some set of values
2. Compute variance σ² over same set
3. Normalize:  x̂ = (x - μ) / √(σ² + ε)
4. Scale + shift:  y = γ·x̂ + β

γ (gamma)   = learnable scale parameter
β (beta)    = learnable shift parameter
ε (epsilon) = small constant (~1e-5) to prevent division by zero
```

---

## BatchNorm — Batch Normalization

**Normalizes across the batch dimension** — for each feature, compute mean and variance across all samples in the batch.

### What it normalizes over

```
Input shape: (B, C, H, W)   CNN       or   (B, D)   MLP
             B = batch size
             C = channels / features
             H, W = height, width

BatchNorm: average over B (and H, W for CNN) for each C independently
```

### Visualization

```
Batch of 4 samples, 3 features:

         Feature 1   Feature 2   Feature 3
Sample 1:   2.1         0.5         3.2
Sample 2:   1.8         0.9         2.7
Sample 3:   2.4         0.3         3.5
Sample 4:   1.9         0.7         2.9

BatchNorm normalizes DOWN each column (across samples per feature)
→ each feature gets mean=0, std=1 across the batch
```

### Formula

```
μ_B  = (1/B) · Σᵢ xᵢ                mean across batch
σ²_B = (1/B) · Σᵢ (xᵢ - μ_B)²      variance across batch

x̂ᵢ  = (xᵢ - μ_B) / √(σ²_B + ε)    normalize
yᵢ   = γ · x̂ᵢ + β                  scale and shift (learned)
```

### Train vs Inference — Critical Difference

During **training:** use current batch statistics.

During **inference:** use **running averages** accumulated during training — because you may predict one sample at a time (batch size = 1 is undefined).

```python
# Running stats updated each batch during training
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var  = (1 - momentum) * running_var  + momentum * batch_var
```

```python
model.train()   # uses batch mean/std     → stochastic
model.eval()    # uses running mean/std   → deterministic
```

Forgetting `model.eval()` during inference is one of the most common bugs in PyTorch. BatchNorm behaves differently depending on mode.

### Implementation

```python
import torch
import torch.nn as nn

# For MLP / fully connected layers
bn = nn.BatchNorm1d(num_features=256)

# For CNN (normalizes across B, H, W per channel)
bn_cnn = nn.BatchNorm2d(num_channels=64)

# What's learned
print(bn.weight.shape)        # torch.Size([256])  →  γ
print(bn.bias.shape)          # torch.Size([256])  →  β
print(bn.running_mean.shape)  # torch.Size([256])  →  accumulated mean
print(bn.running_var.shape)   # torch.Size([256])  →  accumulated variance

# In a model
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.BatchNorm1d(256),      # normalize before activation
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU()
)
```

### Where in the layer

```python
# Original paper: Linear → BatchNorm → Activation
nn.Linear(128, 256)
nn.BatchNorm1d(256)
nn.ReLU()
```

### Problems with BatchNorm

```
Small batch sizes    →  noisy statistics, unstable training
Batch size = 1       →  undefined (std of one number)
RNNs / sequences     →  variable length makes batch stats unreliable
Distributed training →  need to sync batch stats across GPUs
```

**Used in:** ResNet, VGG, most CNN architectures, MLP hidden layers.

---

## LayerNorm — Layer Normalization

**Normalizes across the feature dimension** — for each sample independently, compute mean and variance across all features.

### What it normalizes over

```
Input shape: (B, T, D)   Transformer
             B = batch size
             T = sequence length (tokens)
             D = model dimension (features)

LayerNorm: average over D for each sample and position independently
           B and T are NOT averaged over
```

### Visualization

```
Batch of 4 samples, 3 features:

         Feature 1   Feature 2   Feature 3
Sample 1:   2.1         0.5         3.2      ← normalize ACROSS this row
Sample 2:   1.8         0.9         2.7      ← normalize ACROSS this row
Sample 3:   2.4         0.3         3.5      ← normalize ACROSS this row
Sample 4:   1.9         0.7         2.9      ← normalize ACROSS this row

LayerNorm normalizes ACROSS each row (across features per sample)
→ each sample is independently normalized
```

### Formula

```
μ_L  = (1/D) · Σⱼ xⱼ               mean across features
σ²_L = (1/D) · Σⱼ (xⱼ - μ_L)²     variance across features

x̂ⱼ  = (xⱼ - μ_L) / √(σ²_L + ε)   normalize
yⱼ   = γ · x̂ⱼ + β                 scale and shift (learned)
```

### Why Transformers use LayerNorm instead of BatchNorm

```
Transformers process variable-length sequences
→ batch statistics across variable lengths are unreliable

Each token needs to be independently normalized
→ LayerNorm normalizes each token's features independently

No train / eval discrepancy
→ no running statistics — same behavior always

Works with batch size = 1
→ doesn't depend on batch dimension at all
```

### Implementation

```python
import torch.nn as nn

ln = nn.LayerNorm(normalized_shape=512)  # normalize over last dim

# In a Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-LayerNorm (modern style — GPT-2, LLaMA)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x
```

### Pre-Norm vs Post-Norm

```
# Original Transformer 2017 (Post-Norm)
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))

# Modern Transformers (Pre-Norm) — GPT-2, LLaMA, Mistral
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Pre-Norm is more stable for training very deep Transformers — now the standard.

**Used in:** BERT, GPT-2, GPT-3, all Transformer-based models.

---

## RMSNorm — Root Mean Square Normalization

LayerNorm does two things: re-centering (subtract mean) and re-scaling (divide by std). RMSNorm asks: **do we need the re-centering step?**

The empirical answer: no — re-scaling alone is sufficient, and dropping re-centering makes it faster.

### Formula

```
RMS(x) = √((1/D) · Σⱼ xⱼ²)       root mean square — no mean subtraction

x̂ⱼ  = xⱼ / RMS(x)                normalize — no centering

yⱼ   = γ · x̂ⱼ                    scale only — no β shift parameter
```

**Compared to LayerNorm:**
```
LayerNorm:  subtract mean → divide by std → scale γ → shift β    (4 ops)
RMSNorm:    divide by RMS → scale γ                               (2 ops)
```

### Why It's Faster

```
LayerNorm:  compute mean → compute variance → normalize → scale → shift
RMSNorm:    compute RMS  → normalize → scale

~10–15% faster in practice on large models
```

At the scale of LLaMA-70B running millions of forward passes — this adds up to significant savings.

### Implementation

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))   # γ only, no β

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x   = x / (rms + self.eps)
        return self.scale * x

# PyTorch 2.4+ native
rms_norm = nn.RMSNorm(4096)   # LLaMA-7B uses dim=4096
```

### Where in LLaMA Architecture

```python
class LLaMABlock(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.norm1 = RMSNorm(dim)          # before attention
        self.attn  = Attention(dim, nhead)
        self.norm2 = RMSNorm(dim)          # before FFN
        self.ffn   = SwiGLU_FFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # pre-norm + residual
        x = x + self.ffn(self.norm2(x))   # pre-norm + residual
        return x
```

**Used in:** LLaMA, LLaMA 2, LLaMA 3, Mistral, Gemma, Falcon — current standard for large language models.

---

## Side-by-Side Comparison

### What dimension each normalizes over

```
Input shape: (B=4, T=8, D=512)

BatchNorm:   average over B and T  →  normalize each of D feature columns
             affected by other samples in the batch

LayerNorm:   average over D        →  normalize each B×T position's features
             each token is fully independent

RMSNorm:     average over D (no centering) →  same scope as LayerNorm
             faster — no mean subtraction
```

### Visual

```
           Feature dim (D) →→→→→→→→
         ┌──────────────────────────┐
Batch  B │  ·  ·  ·  ·  ·  ·  ·  · │  ← LayerNorm / RMSNorm normalize ACROSS
(× T)  ↓ │  ·  ·  ·  ·  ·  ·  ·  · │     each row independently
         │  ·  ·  ·  ·  ·  ·  ·  · │
         └──────────────────────────┘
               ↕  ↕  ↕  ↕  ↕  ↕
         BatchNorm normalizes DOWN each column across samples
```

### Comparison Table

| | BatchNorm | LayerNorm | RMSNorm |
|---|---|---|---|
| Normalizes over | Batch + spatial | Feature dim | Feature dim |
| Depends on batch size | Yes | No | No |
| Works with batch size 1 | No | Yes | Yes |
| Train / eval difference | Yes (running stats) | No | No |
| Learnable params | γ and β | γ and β | γ only |
| Speed | Baseline | Similar | ~10–15% faster |
| Used in | CNN · ANN | BERT · GPT-2 | LLaMA · Mistral · Gemma |

---

## Where They Sit in the Full Forward Pass

```
Input x
    ↓
z = W·x + b                         ← Linear Transform
    ↓
z̃ = BatchNorm(z)                    ← CNN / ANN
     or LayerNorm(z)                 ← BERT · GPT-2 · Transformer
     or RMSNorm(z)                   ← LLaMA · Mistral · modern LLMs
    ↓
a = Activation(z̃)                   ← ReLU / GELU / Swish / Tanh
    ↓
a = Dropout(a)                       ← training only
    ↓
Next layer
```

---

## One Line Each

```
BatchNorm   →  normalize across the batch per feature  — needs batch, breaks at size 1
LayerNorm   →  normalize across features per sample    — batch-independent, Transformer standard
RMSNorm     →  LayerNorm without mean centering        — faster, modern LLM standard
```