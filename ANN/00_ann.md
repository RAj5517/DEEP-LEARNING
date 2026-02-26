# ANN — Artificial Neural Network (MLP)

---

## What Is It?

An Artificial Neural Network (ANN), also called a **Multi-Layer Perceptron (MLP)**, is the foundational architecture of deep learning. Every other architecture — CNN, RNN, Transformer — is a specialized extension of this idea.

The core concept: stack layers of **neurons** where each neuron takes a weighted sum of its inputs, adds a bias, passes through an activation function, and sends its output to the next layer. Through training, the network learns the right weights to map inputs to outputs.

It was inspired by biological neurons in the brain, but the connection is loose — it's better understood as a **universal function approximator**: given enough neurons and layers, an MLP can approximate any continuous function to arbitrary precision (Universal Approximation Theorem).

---

## Architecture

### The Single Neuron

```
Inputs:   x₁, x₂, x₃
Weights:  w₁, w₂, w₃
Bias:     b

Output = activation(w₁x₁ + w₂x₂ + w₃x₃ + b)
       = activation(Wx + b)
```

### Full MLP Structure

```
Input Layer        Hidden Layer 1     Hidden Layer 2     Output Layer
                                                         
  x₁ ──────────── n₁₁ ─────────────── n₂₁ ──────────── o₁
  x₂ ──────────── n₁₂ ─────────────── n₂₂ ──────────── o₂
  x₃ ──────────── n₁₃ ─────────────── n₂₃
  x₄ ──────────── n₁₄ ─────────────── n₂₄

  [4 inputs]       [4 neurons]         [4 neurons]        [2 outputs]
  
  Every neuron connects to every neuron in the next layer → "Fully Connected"
```

Each layer performs: `output = activation(W · input + b)`

---

## Key Components

### Activation Functions

Without activation functions, stacking linear layers produces only a linear function — no matter how deep. Activations introduce **nonlinearity**, which is what gives neural networks their expressive power.

```
ReLU:    f(x) = max(0, x)              ← Default for hidden layers
Sigmoid: f(x) = 1 / (1 + e^(-x))      ← Binary classification output
Softmax: f(xᵢ) = e^xᵢ / Σe^xⱼ        ← Multi-class output
Tanh:    f(x) = (e^x - e^(-x))/(e^x + e^(-x))  ← Zero-centered, RNNs
GELU:    smoother ReLU                 ← Transformers, modern networks
```

**ReLU is the default hidden layer activation.** Simple, fast, works well. Its variants (Leaky ReLU, ELU, GELU) address specific weaknesses like the "dying ReLU" problem where neurons output zero for all inputs.

### The Forward Pass

Data flows left to right through the network:

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # Layer 1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),  # Layer 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)   # Output
        )
    
    def forward(self, x):
        return self.net(x)
```

### The Backward Pass — Backpropagation

After the forward pass computes a prediction, the **loss** measures how wrong it was. Backpropagation computes the gradient of the loss with respect to every weight using the chain rule, then the optimizer updates the weights in the direction that reduces loss.

```
Forward:   Input → Predictions → Loss
Backward:  Loss → ∂Loss/∂W for every W → Optimizer updates W
```

This is the core training loop. Every deep learning framework (PyTorch, TensorFlow) handles backprop automatically via autograd.

---

## Architecture Details

### Layer Types in an MLP

**Input layer** — not counted as a "layer" in depth. Just the raw features fed in. No computation happens here.

**Hidden layers** — where learning happens. Each is a Linear transformation followed by activation. More hidden layers = deeper network = can learn more complex patterns.

**Output layer** — no activation (for regression), Sigmoid (binary), or Softmax (multi-class). Size equals number of output targets.

### How Deep Should It Be?

```
1 hidden layer    →  Can approximate most simple functions
2 hidden layers   →  Standard for most tabular tasks
3–5 hidden layers →  Complex tabular, feature learning
Deeper than 5     →  Rarely needed for pure MLP; use CNN/Transformer instead
```

MLP depth is limited by vanishing gradients — unlike CNNs with BatchNorm and Transformers with residuals, plain MLPs struggle to train past ~5–10 layers without special tricks.

### Width (Neurons per Layer)

Common pattern: **funnel** (decreasing) or **constant**:

```
Funnel:    512 → 256 → 128 → 64 → output    (each layer compresses)
Constant:  512 → 512 → 512 → output         (same capacity at each layer)
```

Typical ranges: 64–2048 neurons per layer depending on task complexity.

---

## Loss Functions

The loss function defines what "wrong" means. Chosen based on the task:

| Task | Loss Function | Output Activation |
|------|--------------|-------------------|
| Binary classification | Binary Cross-Entropy | Sigmoid |
| Multi-class classification | Cross-Entropy | Softmax |
| Regression | MSE / MAE | None |
| Multi-label classification | BCE per label | Sigmoid per output |

```python
# Classification
criterion = nn.CrossEntropyLoss()     # Softmax + NLL combined
# Regression  
criterion = nn.MSELoss()
# Binary
criterion = nn.BCEWithLogitsLoss()    # Sigmoid + BCE combined (numerically stable)
```

---

## Where MLP Is Used Today

Pure MLPs are used for **tabular/structured data** — the kind that lives in spreadsheets and databases. For images, audio, and text, specialized architectures (CNN, RNN, Transformer) almost always outperform raw MLPs.

However, MLP components appear **inside every other architecture**:
- The FFN (Feed-Forward Network) block inside every Transformer is a 2-layer MLP
- Classification heads on top of CNNs and Transformers are MLPs
- Actor/critic networks in reinforcement learning are MLPs

Understanding MLP deeply means understanding the building block of all deep learning.

---

## Types / Variants

```
ANN / MLP
│
├── Standard MLP              → Tabular data, classification, regression
│
├── Autoencoder               → Unsupervised, dimensionality reduction
│     ├── Vanilla Autoencoder
│     ├── Variational (VAE)   → Generative modeling
│     └── Denoising           → Robust representations
│
├── Siamese Network           → Similarity learning, few-shot
│
└── Modern MLP Variants
      ├── MLP-Mixer           → Pure MLP for vision (patch mixing)
      └── TabNet              → Attention-based MLP for tabular data
```

For the standard MLP variants (Autoencoder, Siamese), see `ann_variants.md`.

---

## Quick Reference

```python
# Standard training setup for tabular classification
model = MLP(input_dim=20, hidden_dim=256, output_dim=5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**When to use MLP:**
- Input is tabular / structured (rows and columns)
- No spatial structure (images → CNN), no sequence (text → Transformer)
- As a head on top of any other architecture for final classification/regression