# Linear Transform — W·x + b

---

## What It Is

Every neuron in a neural network does one thing at its core — a linear transformation. This is the actual computation that happens before any activation function.

```
output = W·x + b

W  = weight matrix   (learned parameters)
x  = input vector    (data coming in)
b  = bias vector     (learned parameters)
·  = matrix multiplication
```

---

## Single Neuron vs Full Layer

**Single neuron:**
```
x = [x₁, x₂, x₃]              input: 3 features
w = [w₁, w₂, w₃]              weights: one per input
b = scalar bias

output = w₁x₁ + w₂x₂ + w₃x₃ + b   →  single number
```

**Full layer (multiple neurons):**
```
x  shape: (3,)        →  3 input features
W  shape: (4, 3)      →  4 neurons, each with 3 weights
b  shape: (4,)        →  one bias per neuron

output = W·x + b      →  shape (4,)   →  4 output values
```

**Batched (real training):**
```
X  shape: (32, 3)     →  batch of 32 samples, 3 features each
W  shape: (3, 4)      →  weight matrix
b  shape: (4,)        →  bias

output = X·W + b      →  shape (32, 4)
```

---

## What W and b Actually Do

### Weight Matrix W

Controls how much each input contributes to each output neuron.

```
W = [[0.5, -0.3,  0.8],    ← neuron 1: weights for x₁, x₂, x₃
     [0.2,  0.9, -0.1],    ← neuron 2
     [-0.4, 0.1,  0.6],    ← neuron 3
     [0.7, -0.5,  0.3]]    ← neuron 4
```

```
Large positive weight  →  input strongly activates the neuron
Large negative weight  →  input strongly suppresses the neuron
Weight near 0          →  input is mostly ignored
```

### Bias b

Shifts the output — allows the neuron to fire even when all inputs are 0. Without bias, every decision boundary must pass through the origin.

```
Without bias:   output = W·x       (forced through origin)
With bias:      output = W·x + b   (can shift anywhere)
```

---

## Implementation

```python
import torch
import torch.nn as nn

# Single linear layer
linear = nn.Linear(in_features=3, out_features=4)

# What's inside
print(linear.weight.shape)   # torch.Size([4, 3])  →  W
print(linear.bias.shape)     # torch.Size([4])     →  b

# Forward pass
x = torch.randn(32, 3)       # batch of 32 samples
output = linear(x)
print(output.shape)          # torch.Size([32, 4])

# Manual equivalent — identical result
output_manual = x @ linear.weight.T + linear.bias
```

---

## Weight Initialization — Why It Matters Enormously

If weights start too large → activations explode → gradients explode → training diverges.
If weights start too small → activations vanish → gradients vanish → training stalls.

The goal: keep the variance of activations roughly equal across all layers.

### Xavier / Glorot Initialization

Designed for Sigmoid and Tanh activations.

```
Formula:
W ~ Uniform(-√(6/(nᵢₙ + nₒᵤₜ)), √(6/(nᵢₙ + nₒᵤₜ)))
```

```python
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

### He / Kaiming Initialization

Designed for ReLU and its variants. Accounts for the fact that ReLU kills half the neurons (negative values → 0), so variance needs to be doubled.

```
Formula:
W ~ Normal(0, √(2/nᵢₙ))
```

```python
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

`nn.Linear` uses Kaiming Uniform by default in PyTorch.

### Initialization Guide

```
Sigmoid / Tanh  →  Xavier (Glorot)
ReLU family     →  He (Kaiming)       ← PyTorch default
Transformers    →  Xavier or custom small init
LSTM            →  Orthogonal for recurrent weights
```

---

## The Full Forward Pass for One Layer

```
Input x
    ↓
z = W·x + b          ← Linear Transform  (pre-activation value)
    ↓
z̃ = Normalize(z)     ← BatchNorm / LayerNorm / RMSNorm
    ↓
a = Activation(z̃)    ← ReLU / GELU / Tanh etc.
    ↓
a = Dropout(a)        ← training only
    ↓
Output to next layer
```

The linear transform creates the pre-activation value `z` — also called **logit** or **pre-activation**. Everything after it shapes how that value flows forward.

---

## Why Linear Transform Alone Is Not Enough

Stack 3 linear layers without activation:

```
z₁ = W₁·x + b₁
z₂ = W₂·z₁ + b₂  =  W₂·(W₁·x + b₁) + b₂
z₃ = W₃·z₂ + b₃  =  (W₃W₂W₁)·x + constant
```

This is just one linear transform with combined matrix `W₃W₂W₁`. A 100-layer network without activations is identical to a single-layer network. Depth is meaningless without non-linearity between layers.

---

## Summary

```
W·x      →  weighted sum of inputs  (how much each input matters)
  + b    →  bias shift              (allows output when all inputs are 0)
= z      →  pre-activation logit    (raw signal before shaping)
```

Linear transform is the engine. Everything else — normalization, activation, dropout — shapes what comes out of it.