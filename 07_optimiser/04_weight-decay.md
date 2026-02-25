# Weight Decay & L2 Regularization

---

## What Is Weight Decay?

As you train a neural network, the optimizer's only goal is to minimize the training loss. Left unconstrained, it will find any set of parameters that achieves low training loss — including ones that are extremely large in magnitude and highly specialized to the training data. This is called **overfitting**: the model memorizes the training set instead of learning generalizable patterns.

**Weight decay** is a regularization technique that adds pressure to keep the parameter values **small**. The intuition: small weights correspond to simpler, smoother functions. A network with small weights can't make extremely sharp, overfit decisions — it's forced to find patterns that are robust and general.

Weight decay is arguably the single most important and most universally applied regularization technique in deep learning. Unlike dropout or batch normalization (which have specific architectural implications), weight decay is a **pure optimizer-level concept** that applies to any model.

---

## L2 Regularization: The Mathematical Foundation

The standard formulation adds a penalty term to the loss function that grows with the magnitude of the weights:

```
L_regularized(θ) = L(θ) + (λ/2) · ||θ||²
```

Where:
- `L(θ)` = original training loss (cross-entropy, MSE, etc.)
- `λ` = weight decay coefficient (controls strength of regularization)
- `||θ||²` = sum of squares of all parameters = `Σ θᵢ²`

The gradient of this regularized loss is:

```
∇L_regularized(θ) = ∇L(θ) + λ·θ
```

The extra term `λ·θ` is just the current parameter value scaled by `λ`. Adding this to the gradient means: even if `∇L(θ) = 0` (we're at a loss minimum), the gradient is still `λ·θ`, pushing the parameters toward zero.

The SGD update with L2 regularization becomes:

```
θ_t = θ_{t-1} - η · (∇L(θ_{t-1}) + λ·θ_{t-1})
     = θ_{t-1} · (1 - η·λ) - η · ∇L(θ_{t-1})
```

The factor `(1 - η·λ)` **directly shrinks the weights** at every step — hence the name "weight decay."

**With SGD, L2 regularization and weight decay are mathematically identical.** This is a crucial fact that leads to a common misconception with Adam.

---

## The Adam Trap: Why `weight_decay` in Adam Is Not L2 Regularization

This is one of the most important subtle points in deep learning optimization, and it's widely misunderstood.

When you add `weight_decay` to Adam's `torch.optim.Adam`, the implementation adds `λ·θ` to the gradient *before* the adaptive scaling step:

```python
# What torch.optim.Adam does with weight_decay:
g_t = ∇L(θ_t) + λ·θ_t          # Add L2 penalty to gradient

# Then proceed with Adam update:
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

The regularization term `λ·θ_t` gets added to the gradient and then **divided by `√v̂_t`**, the adaptive scaling factor. This means:

- Parameters with large gradient variance (large `√v̂`) → decay effect is weakened
- Parameters with small gradient variance (small `√v̂`) → decay effect is amplified

The effective weight decay is no longer uniform — it varies per parameter based on gradient history. This is **not L2 regularization**. It's some other, inconsistent form of regularization that:
1. Is theoretically poorly motivated
2. Tends to under-regularize parameters with high gradient variance
3. Does not correspond to the intuition behind weight decay

### The Fix: AdamW (Decoupled Weight Decay)

AdamW applies the weight decay **directly to the parameters**, bypassing the adaptive scaling:

```python
# AdamW:
g_t = ∇L(θ_t)                           # Pure gradient (no L2 added)
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε) - η · λ · θ_{t-1}   # Decay applied separately
```

Now the weight decay term `- η · λ · θ_{t-1}` is the same for all parameters regardless of their gradient history. This is proper, decoupled weight decay.

**Practical implication:** If you use `torch.optim.Adam(weight_decay=0.01)`, the regularization is broken. Use `torch.optim.AdamW(weight_decay=0.01)` instead. For SGD, the two are equivalent — `torch.optim.SGD(weight_decay=0.01)` is fine.

---

## How Much Weight Decay to Use?

The `λ` parameter is one of the most important hyperparameters to tune. Here are empirically established ranges:

### For SGD + Momentum (Computer Vision)

```python
# Standard ResNet/ImageNet training
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
```

| Architecture | Task | Typical weight_decay |
|-------------|------|---------------------|
| ResNet | ImageNet | 1e-4 |
| EfficientNet | ImageNet | 1e-5 |
| VGG | ImageNet | 5e-4 |
| MobileNet | ImageNet | 1e-4 to 4e-5 |

**Vision rule of thumb:** `1e-4` is a safe default for most CNNs from scratch.

### For AdamW (Transformers / NLP)

```python
# BERT fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# GPT-style pretraining
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
```

| Model | Task | Typical weight_decay |
|-------|------|---------------------|
| BERT fine-tuning | Classification | 0.01 |
| GPT-2/3 style | Pretraining | 0.1 |
| ViT | ImageNet | 0.05 to 0.3 |
| CLIP | Pretraining | 0.2 |

**Transformer rule of thumb:** Much higher weight decay than CNNs — typically 0.01 to 0.1. Transformers with AdamW can absorb stronger regularization.

### General Guidance

- **Too small** (λ → 0): No regularization, model may overfit on small datasets
- **Too large** (λ → 1): Model is over-regularized, underfits, weights collapse toward zero
- The sweet spot depends on **dataset size**: larger datasets can use smaller λ (less regularization needed); smaller datasets need larger λ

---

## What to Apply Weight Decay To

A critical detail: **not all parameters should be decayed.** The standard practice is:

### Parameters to DECAY (apply weight_decay)
- Weight matrices in linear/attention/conv layers
- The reason: these are the parameters that can grow large and cause overfitting

### Parameters NOT to decay (set weight_decay=0)
- **Biases** — biases are scalar offsets that rarely cause overfitting; decaying them introduces unnecessary bias toward zero
- **LayerNorm / BatchNorm weights and biases** — these are scaling/shifting parameters for normalization layers; decaying them interferes with the normalization mechanism
- **Embedding positions** — in some architectures

```python
# The standard implementation of selective weight decay
def get_parameter_groups(model, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'ln_f.weight']
    
    decay_params = [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad]
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

optimizer = torch.optim.AdamW(
    get_parameter_groups(model, weight_decay=0.01),
    lr=2e-5
)
```

This pattern appears in virtually all serious transformer training code.

---

## Weight Decay vs. Other Regularization Techniques

Weight decay doesn't exist in a vacuum. Here's how it interacts with other regularization methods:

| Technique | What It Does | Interacts with Weight Decay? |
|-----------|-------------|------------------------------|
| **Weight Decay** | Penalizes large weights | — |
| **Dropout** | Randomly zeros activations during training | Complementary; both can be used |
| **BatchNorm** | Normalizes layer inputs | Somewhat redundant — BN reduces the need for heavy WD |
| **Data Augmentation** | Increases effective dataset size | Complementary |
| **Early Stopping** | Stops training when val loss increases | Alternative to WD for overfitting control |
| **Label Smoothing** | Softens target distribution | Complementary |

For transformers, the typical combination is: **AdamW + weight decay + dropout + data augmentation**. BatchNorm is rarely used in transformers (LayerNorm is preferred), so weight decay does more regularization work.

---

## Practical Examples

### Training ResNet-50 on CIFAR-10 (Small Dataset → More Regularization)

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,  # Slightly higher than ImageNet because dataset is smaller
    nesterov=True
)
```

### Fine-tuning BERT (Medium Dataset)

```python
param_groups = get_parameter_groups(model, weight_decay=0.01)
optimizer = torch.optim.AdamW(param_groups, lr=2e-5)
```

### Training GPT-2 from Scratch

```python
param_groups = get_parameter_groups(model, weight_decay=0.1)
optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))
# Note: (0.9, 0.95) for β instead of (0.9, 0.999) is common for LLM pretraining
```

---

## Key Takeaways

- Weight decay is **essential regularization** for almost every deep learning model — it should rarely be set to zero.
- **L2 regularization and weight decay are equivalent for SGD** but NOT for Adam. Use AdamW for correct weight decay with adaptive optimizers.
- **Don't decay biases and norm parameters** — separate your parameter groups.
- Vision models typically use **1e-4 to 5e-4** with SGD; transformers typically use **0.01 to 0.1** with AdamW.
- Higher weight decay is needed for smaller datasets; larger datasets can tolerate smaller values.