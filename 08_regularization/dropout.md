# Dropout & Variants — Regularization During the Forward Pass

---

## What Is Dropout (and Why Does Regularization Happen Here)?

Most regularization techniques act on the **optimizer** (weight decay) or the **training loop** (early stopping). Dropout is different — it injects randomness directly into the **forward pass**, changing the computation itself during training.

The core idea: at every training step, randomly **zero out a fraction of neurons** (or feature maps, or tokens, depending on the variant). This forces the network to learn **redundant, distributed representations** — it cannot rely on any single neuron always being present, so every neuron must learn to be useful on its own and in combination with many different subsets of other neurons.

The conceptual framing that made dropout famous: training a network with dropout is approximately like training an **exponential ensemble** of `2^n` different sub-networks (where `n` is the number of neurons) that share weights, and averaging their predictions at test time. This ensemble effect is a powerful form of regularization.

Without dropout, a neuron can develop **co-adaptation** — it learns to fix the errors of specific other neurons, becoming dependent on their presence. This is a form of overfitting at the representational level. Dropout breaks co-adaptation by making the presence of any neuron unreliable.

---

## Standard Dropout ⭐ Most Important

### The Mechanism

During **training**: for each forward pass, independently zero each neuron's output with probability `p` (the dropout rate). The surviving neurons are scaled up by `1/(1-p)` to keep the expected value unchanged.

```
# For each neuron activation h:
mask ~ Bernoulli(1 - p)    # 1 with prob (1-p), 0 with prob p
h_dropped = h * mask / (1 - p)   # Inverted dropout (modern standard)
```

The `1/(1-p)` scaling is called **inverted dropout** — scaling happens during training so that at test time, no modification is needed (just use the full network as-is). This is what PyTorch implements.

During **inference**: dropout is disabled entirely. The full network is used with all neurons active. This is why you must call `model.eval()` before evaluating — it disables dropout (and batch norm training mode).

### The Math of the Ensemble Interpretation

With `n` neurons and dropout rate `p`, there are `2^n` possible sub-networks. In practice, at each forward pass you sample one of these uniformly. Over many training steps, you've approximately trained all `2^n` networks with shared weights. At test time, using the full network with scaled weights approximates taking the geometric mean of all `2^n` networks' predictions — this is the ensemble effect.

For `n=1000` neurons, that's `2^1000` networks. No other ensemble method is computationally feasible at this scale.

### Dropout Rate: How Much to Drop?

The rate `p` is the most important hyperparameter:

| Layer Type | Typical `p` | Notes |
|-----------|------------|-------|
| Fully connected (hidden) | 0.5 | The classic Hinton value |
| Fully connected (small networks) | 0.2–0.3 | Less aggressive |
| Convolutional layers | 0.1–0.2 | Rarely applied; spatial dropout preferred |
| Input layer | 0.1–0.2 | Only in specific cases |
| Transformer (attention/FFN) | 0.1 | Lower rates for transformers |
| LSTM/GRU | 0.2–0.5 | Applied to inputs/outputs, not recurrent connections |

`p = 0.5` maximizes the number of possible sub-networks (ensemble size) but is too aggressive for small networks or convolutional layers where spatial correlations matter.

### PyTorch Usage

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),      # Applied AFTER activation
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.net(x)

model = MLP()

# CRITICAL: switch modes correctly
model.train()   # Dropout ACTIVE
output = model(x)

model.eval()    # Dropout DISABLED
with torch.no_grad():
    prediction = model(x)
```

A very common bug: forgetting `model.eval()` at inference time, making predictions non-deterministic and wrong.

### Where to Place Dropout

**Place dropout AFTER the activation function**, not before. The reasoning: dropout zeros outputs, so applying it after the activation preserves the statistical properties of the activation distribution.

```python
# Correct order:
nn.Linear(512, 256) → nn.ReLU() → nn.Dropout(0.5)

# Incorrect (technically works but less principled):
nn.Linear(512, 256) → nn.Dropout(0.5) → nn.ReLU()
```

**Don't apply dropout to the output layer** — you want deterministic logits.

**Don't apply dropout between BatchNorm and what follows it** — BatchNorm and Dropout interact badly. Dropout randomly zeros activations, which distorts the mean and variance statistics that BatchNorm tracks. Place dropout before BatchNorm, or avoid the combination entirely. In practice, most modern architectures use one or the other, not both.

### Dropout in Practice: When It Helps and When It Doesn't

**Where dropout works well:**
- Large fully-connected layers in classification heads
- LSTM/GRU recurrent networks
- Small-to-medium scale models with risk of overfitting
- Transformer models (with low dropout rates ~0.1)

**Where dropout doesn't work well:**
- **Convolutional layers with small kernels**: spatially adjacent pixels are highly correlated, so dropping individual activations doesn't prevent co-adaptation — neurons can simply learn from neighboring surviving neurons. Use Spatial Dropout instead.
- **Small datasets with tiny models**: the regularization is too aggressive; early stopping or weight decay is more appropriate
- **Batch sizes of 1**: Batch normalization is already stochastic at small batch sizes; adding dropout is redundant noise

### Dropout for Uncertainty Estimation (MC Dropout)

A non-obvious application: keep dropout enabled at **test time** and run the forward pass multiple times. The variance across outputs gives an estimate of **model uncertainty** (epistemic uncertainty). This is called Monte Carlo Dropout and is used in production for uncertainty-aware predictions:

```python
model.train()   # Keep dropout ON intentionally

n_samples = 50
predictions = torch.stack([model(x) for _ in range(n_samples)])

mean_prediction = predictions.mean(dim=0)
uncertainty = predictions.std(dim=0)    # High std = model is uncertain
```

---

## Spatial Dropout (Dropout2D)

Standard dropout zeroes individual activations. For convolutional networks, this is ineffective because neighboring activations are correlated — zeroing one pixel barely disrupts the learned feature maps.

**Spatial Dropout** (also called `Dropout2D` in PyTorch) instead zeroes entire **feature maps** (channels). If a channel is dropped, all spatial positions in that channel are zeroed. This is a much stronger signal because the entire learned feature is removed, forcing the network to be redundant across channels.

```python
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 128, 3, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.dropout2d = nn.Dropout2d(p=0.2)   # Drop entire channels

    def forward(self, x):
        return self.dropout2d(F.relu(self.bn(self.conv(x))))
```

Spatial dropout is significantly more effective than standard dropout for convolutional architectures.

---

## DropPath (Stochastic Depth)

DropPath, introduced in the **Stochastic Depth** paper (Huang et al., 2016) and popularized by vision transformers (DeiT, Swin, ConvNeXt), drops entire **residual branches** rather than individual neurons.

In a residual network:
```
# Standard residual:
output = x + F(x)    # F is the residual branch (conv/attention layers)

# With DropPath:
output = x + DropPath(F(x))   # Entire branch randomly zeroed
```

During training, each layer's residual branch is independently dropped with probability `p`. This means some forward passes skip layers entirely — effectively training an ensemble of networks with different depths.

**Why it works so well for transformers:**
- Transformers are very deep (12–96+ layers) and co-adaptation across layers is a real problem
- Dropping entire layers is a much stronger and more natural form of regularization than dropping individual neurons
- It also acts as an implicit depth reduction, giving the model "shorter path" training signal

```python
from timm.models.layers import DropPath

class TransformerBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim)
        self.drop_path = DropPath(drop_path_rate)   # From timm
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
```

DropPath rate is typically increased with depth — shallow layers get lower rates, deep layers get higher rates. This is called **linear rate scheduling** and is used in Swin Transformer, DeiT, ConvNeXt:

```python
# Linear rate schedule across layers
num_layers = 12
max_drop_path_rate = 0.2
drop_path_rates = [x.item() for x in torch.linspace(0, max_drop_path_rate, num_layers)]
```

---

## DropConnect (Brief Introduction)

DropConnect (Wan et al., 2013) is a generalization of dropout: instead of zeroing neuron **outputs**, it zeros individual **weights** in the weight matrix during training. Each weight connection is independently dropped with probability `p`.

Theoretically more general than dropout, but computationally more expensive and doesn't show consistent improvements. Rarely used in practice today.

---

## Comparison: Which Dropout Variant to Use?

| Variant | Applied To | Best For |
|---------|-----------|----------|
| Standard Dropout | Individual activations | FC layers, transformers, RNNs |
| Spatial Dropout (2D) | Entire feature maps (channels) | CNN feature extraction |
| DropPath | Entire residual branches | Transformers, deep ResNets |
| DropConnect | Individual weights | Rarely used; theoretical interest |

---

## Key Takeaways

- **Standard Dropout (p=0.5)** is the default for fully-connected layers. Lower rates (0.1–0.2) for transformers and convolutional architectures.
- **Always call `model.eval()`** at inference time — forgetting this is one of the most common bugs in PyTorch.
- **Don't mix Dropout and BatchNorm** in the same block — they interact poorly.
- **Spatial Dropout** is the right choice for convolutional networks, not standard dropout.
- **DropPath** is the standard for vision transformers (DeiT, Swin, ConvNeXt) and is more effective than standard dropout for deep residual architectures.
- Dropout rates are typically higher for larger models and smaller datasets — more capacity and less data both increase overfitting risk.