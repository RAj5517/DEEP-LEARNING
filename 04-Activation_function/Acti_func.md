# Activation Functions — Complete Guide

---

## Why Activation Functions Exist

Without activation functions, a neural network collapses into a single linear transformation no matter how many layers you stack:

```
Layer 1: a = W₁·x
Layer 2: a = W₂·(W₁·x) = (W₂W₁)·x
Layer 3: a = W₃·(W₂W₁)·x = (W₃W₂W₁)·x
```

A 100-layer network without activations is identical to a single-layer network.

**With activation functions:**
```
Layer 1: a = ReLU(W₁·x)      ← non-linear
Layer 2: a = ReLU(W₂·a)      ← non-linear on top of non-linear
Layer 3: a = ReLU(W₃·a)      ← genuinely new representation
```

Now depth creates expressive power. The network can approximate any function.

### What activation functions do

```
1. Introduce non-linearity    → learn curves, complex boundaries, not just lines
2. Control information flow   → decide what passes forward and how
```

### The gradient problem — why this all matters

Every activation is a trade-off between:

```
Non-linearity    — can it help the model learn complex patterns?
Gradient flow    — does backprop work well through it?
Compute cost     — how expensive is forward + backward?
Output range     — does it cause exploding or vanishing signals?
```

**The evolution:**
```
Sigmoid / Tanh    →  vanishing gradients in deep nets
     ↓
ReLU              →  fixed vanishing, introduced dead neurons
     ↓
Leaky ReLU / ELU  →  fixed dead neurons
     ↓
GELU / Swish      →  smooth, probabilistic gating, best empirical results
     ↓
SwiGLU            →  gated FFN for LLMs, current state of the art
```

---

## Quick Reference Card

```
Task / Location           →  Use This

Hidden layer (CNN / MLP)  →  ReLU
Hidden layer (modern)     →  GELU or Swish
Dead neurons problem      →  Leaky ReLU or ELU
Transformer hidden        →  GELU
LLM FFN block             →  SwiGLU
Binary output             →  Sigmoid
Multi-class output        →  Softmax
LSTM / RNN gates          →  Tanh
Smoother than ReLU        →  ELU
Self-normalizing MLP      →  SELU
Learnable slope           →  PReLU
PyTorch NLL loss          →  LogSoftmax
```

---

## ✅ Must Know — The 8

---

### 1. ReLU — Rectified Linear Unit

**Formula:**
```
f(x) = max(0, x)

x < 0  →  0
x ≥ 0  →  x (unchanged)
```

**Shape:**
```
         /
        /
_______/
      0
```

**Why it works:**
- Computationally trivial — just a threshold
- Sparse activation — roughly half the neurons output 0, making the network efficient
- No vanishing gradient for positive values — gradient is exactly 1

**The dead neuron problem:**
If a neuron's input is always negative, it always outputs 0, gradient is always 0, weights never update. That neuron is permanently dead.

```python
import torch
import torch.nn as nn

relu = nn.ReLU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(relu(x))
# tensor([0., 0., 0., 1., 2.])
```

**Gradient:**
```
x > 0  →  1
x < 0  →  0  (dead zone)
```

**Used in:** CNN (ResNet, VGG), MLP hidden layers — default choice for most non-transformer models.

---

### 2. Leaky ReLU

**Formula:**
```
f(x) = x        if x ≥ 0
f(x) = 0.01·x   if x < 0

α = 0.01 (small negative slope — the "leak")
```

**Shape:**
```
         /
        /
  ....../   (tiny slope on left, not flat)
      0
```

**Why it exists:**
Fixes the dead neuron problem. Negative inputs still get a small gradient (0.01) so weights can still update even when input is negative.

**Gradient:**
```
x ≥ 0  →  1
x < 0  →  0.01  (not zero — neuron stays alive)
```

```python
leaky = nn.LeakyReLU(negative_slope=0.01)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(leaky(x))
# tensor([-0.0200, -0.0100,  0.0000,  1.0000,  2.0000])
```

**Used in:** GANs (very common), any model where dead neurons are a problem.

---

### 3. ELU — Exponential Linear Unit

**Formula:**
```
f(x) = x              if x ≥ 0
f(x) = α(eˣ - 1)      if x < 0

α = 1.0 (default)
```

**Shape:**
```
         /
        /
~~~~~~~/    (smooth curve into negative, not sharp corner)
      0
```

**Why better than Leaky ReLU:**
- Smooth curve for negatives — no sharp corner at 0
- Negative outputs push mean activation toward 0 — faster convergence
- Saturates at `-α` for very negative values — noise robust

**Gradient:**
```
x ≥ 0  →  1
x < 0  →  α·eˣ   (always positive, smooth)
```

```python
elu = nn.ELU(alpha=1.0)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(elu(x))
# tensor([-0.8647, -0.6321,  0.0000,  1.0000,  2.0000])
```

**Used in:** When you want ReLU benefits + smoother gradients + zero-centered outputs.

---

### 4. GELU — Gaussian Error Linear Unit

**Formula:**
```
f(x) = x · Φ(x)

Φ(x) = cumulative distribution function of standard normal
      = probability that a standard normal value ≤ x

Approximation (used in practice):
f(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
```

**Shape:**
```
         /
        /
  ~~~~~    (slight dip below 0 near x = -0.17, then recovers)
      0
```

**The intuition:**
- ReLU: "if x > 0, pass it. If x < 0, kill it." Hard binary decision.
- GELU: "if x is very positive, pass it. Slightly positive, pass most. Slightly negative, pass a little. Very negative, kill it." Soft probabilistic gating.

**Why Transformers use it:**
- Smooth everywhere — no sharp corners
- Stochastic interpretation — gates each neuron by probability it would be kept under Gaussian noise
- Empirically outperforms ReLU on NLP tasks consistently
- Slight negative region helps with gradient flow

```python
gelu = nn.GELU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(gelu(x))
# tensor([-0.0454, -0.1587,  0.0000,  0.8413,  1.9546])
```

**Used in:** BERT, GPT-2, GPT-3, GPT-4, all modern Transformers — the standard for attention-based models.

---

### 5. Swish / SiLU

**Formula:**
```
f(x) = x · σ(x)

σ(x) = sigmoid(x) = 1 / (1 + e⁻ˣ)
```

**Shape:** Almost identical to GELU — smooth, slight negative dip, unbounded positive.

**Why it works:**
- Self-gated — the input gates itself through sigmoid
- Non-monotonic — slight dip helps gradient flow
- Discovered by Google Brain via neural architecture search (not hand-designed)
- Smoother than ReLU, faster to compute than GELU

**Swish vs GELU:**
They look nearly identical in practice. GELU uses Gaussian CDF, Swish uses sigmoid. Swish is slightly faster. Both outperform ReLU on most tasks.

```python
silu = nn.SiLU()   # PyTorch calls Swish "SiLU"
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(silu(x))
# tensor([-0.2384, -0.2689,  0.0000,  0.7311,  1.7616])
```

**Used in:** EfficientNet, MobileNetV3, modern CNN architectures, some LLMs.

---

### 6. Sigmoid

**Formula:**
```
f(x) = 1 / (1 + e⁻ˣ)

Output range: (0, 1)
```

**Shape:**
```
    ___________  ← 1
   /
  /
 /
/___________     ← 0
```

**The vanishing gradient problem:**
```
Gradient = f(x) · (1 - f(x))

At x = 0:    gradient = 0.25   (maximum)
At x = 2:    gradient = 0.10
At x = 5:    gradient = 0.006
At x = 10:   gradient ≈ 0.000045
```

For very large or very small inputs, gradient is nearly zero. Across many layers during backprop — gradient vanishes completely. This is why sigmoid was abandoned for hidden layers in deep networks.

```python
sigmoid = nn.Sigmoid()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(sigmoid(x))
# tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
```

**Used in:** Binary classification output layer only. Not in hidden layers of deep networks.

---

### 7. Tanh

**Formula:**
```
f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

Output range: (-1, 1)
```

**Shape:**
```
    ___________  ← +1
   /
  /
 /
/___________     ← -1
```

**Why better than sigmoid (historically) for hidden layers:**
- Zero-centered — outputs around 0, not 0.5
- Zero-centered outputs mean gradients in next layer are more balanced
- Still has vanishing gradient but less severe than sigmoid

**Gradient:**
```
gradient = 1 - f(x)²

At x = 0:  gradient = 1.0   (maximum)
At x = 2:  gradient = 0.07
At x = 3:  gradient = 0.01
```

```python
tanh = nn.Tanh()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(tanh(x))
# tensor([-0.9640, -0.7616,  0.0000,  0.7616,  0.9640])
```

**Why LSTM uses tanh specifically:**
Cell state values need to be centered around 0 so they can grow and shrink symmetrically. Tanh's (-1, 1) range is perfect for this.

**Used in:** LSTM gates, GRU gates, RNN hidden states — still the standard inside recurrent architectures.

---

### 8. Softmax

**Formula:**
```
f(xᵢ) = eˣⁱ / Σⱼ eˣʲ

For a vector [x₁, x₂, x₃]:
output = [eˣ¹/sum, eˣ²/sum, eˣ³/sum]

Properties:
- All outputs between 0 and 1
- All outputs sum to exactly 1
- Largest input gets the most probability
```

**Why it works for classification:**
Converts raw logits (any real numbers) into a proper probability distribution. The model is trained with Cross Entropy loss against one-hot targets.

**Temperature scaling:**
```
f(xᵢ) = eˣⁱ/ᵀ / Σⱼ eˣʲ/ᵀ

T < 1  →  sharper (more confident)
T > 1  →  flatter  (more uncertain)
T = 1  →  standard softmax
```

Temperature scaling is used during LLM inference to control randomness.

```python
softmax = nn.Softmax(dim=-1)
logits = torch.tensor([2.0, 1.0, 0.5])
print(softmax(logits))
# tensor([0.5954, 0.2430, 0.1616])  ← sums to 1.0
```

**Used in:** Multi-class classification output. Also used inside Transformers — attention scores are literally softmax over dot products.

---

## ⚠️ Learn When You See It

---

### SELU — Scaled ELU

**Formula:**
```
f(x) = λ · x              if x > 0
f(x) = λ · α(eˣ - 1)      if x ≤ 0

λ = 1.0507   α = 1.6733   (mathematically derived — not arbitrary)
```

**The big idea:** Self-normalizing. If weights are initialized with LeCun normal init, SELU networks maintain `mean ≈ 0` and `std ≈ 1` through all layers automatically — no BatchNorm needed.

**When to use:** Deep MLPs where you want self-normalization without explicit normalization layers. Rarely used today because BatchNorm + ReLU/GELU works better in practice.

---

### PReLU — Parametric ReLU

**Formula:**
```
f(x) = x       if x ≥ 0
f(x) = α·x     if x < 0

α is a learnable parameter (not fixed like Leaky ReLU's 0.01)
```

The model learns the best negative slope during training.

```python
prelu = nn.PReLU()   # α initialized to 0.25, learned during training
```

More expressive than Leaky ReLU, slightly more parameters. Used when you want the model to decide how much negative signal to pass.

---

### LogSoftmax

**Formula:**
```
f(xᵢ) = log(softmax(xᵢ)) = xᵢ - log(Σⱼ eˣʲ)
```

Numerically more stable than computing softmax then taking log separately.

```python
# These two are equivalent:
loss = nn.CrossEntropyLoss()(logits, targets)

log_probs = nn.LogSoftmax(dim=-1)(logits)
loss = nn.NLLLoss()(log_probs, targets)
```

Used with `nn.NLLLoss()` in PyTorch. Know this when you see NLLLoss in code.

---

### SwiGLU

**Formula:**
```
f(x) = Swish(Wx) · Vx

Two linear projections:
- one gated by Swish
- one linear
Then elementwise multiply
```

**Why LLMs use it:**
In the FFN block of Transformers, SwiGLU replaces the standard `Linear → Activation → Linear` pattern with a gated version. More expressive with the same parameter count.

**Used in:** LLaMA, PaLM, Mistral, Gemma — current standard for large language model FFN blocks.

---

### GLU Family

Gated Linear Units — a general pattern where one linear projection gates another:

```
GLU:    f(x) = Wx · σ(Vx)           gate = sigmoid
GeGLU:  f(x) = Wx · GELU(Vx)        gate = GELU
SwiGLU: f(x) = Wx · Swish(Vx)       gate = Swish  ← most used in LLMs
ReGLU:  f(x) = Wx · ReLU(Vx)        gate = ReLU
```

The gating mechanism gives the network more control over what information flows through the FFN layer.

---

### Mish

**Formula:**
```
f(x) = x · tanh(softplus(x))
     = x · tanh(ln(1 + eˣ))
```

Smooth, non-monotonic, unbounded above, slightly negative below — similar to Swish/GELU in behavior. Empirically better than ReLU/Swish in some vision tasks.

**Used in:** YOLOv4, some vision research papers.

---

## Comparison Table

| Activation | Range | Non-monotonic | Vanishing Grad | Dead Neurons | Used In |
|---|---|---|---|---|---|
| ReLU | [0, ∞) | No | No | Yes | CNN, MLP |
| Leaky ReLU | (-∞, ∞) | No | No | No | GANs |
| ELU | (-α, ∞) | No | No | No | General |
| GELU | ~(-0.17, ∞) | Yes | No | No | Transformers |
| Swish/SiLU | ~(-0.28, ∞) | Yes | No | No | Vision, LLMs |
| Sigmoid | (0, 1) | No | Yes | No | Binary output |
| Tanh | (-1, 1) | No | Partial | No | LSTM/RNN |
| Softmax | (0, 1) | No | No | No | Multi-class output |
| SELU | (-λα, ∞) | No | No | No | Self-norm MLP |
| PReLU | (-∞, ∞) | No | No | No | Learnable slope |
| SwiGLU | (-∞, ∞) | Yes | No | No | LLM FFN |
| Mish | ~(-0.31, ∞) | Yes | No | No | Vision research |