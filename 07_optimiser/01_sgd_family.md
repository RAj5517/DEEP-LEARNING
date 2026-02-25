# SGD Family — Stochastic Gradient Descent & Variants

---

## What Is the SGD Family?

At its core, every optimizer in the SGD family shares one unifying idea: **move the parameters in the direction that reduces the loss, using gradients as a compass.**

The classical update rule is:

```
θ ← θ - η · ∇L(θ)
```

Where:
- `θ` = model parameters
- `η` = learning rate
- `∇L(θ)` = gradient of the loss with respect to parameters

The "Stochastic" in SGD means we compute this gradient not over the entire dataset (that would be *Batch Gradient Descent*) but over a **random mini-batch**. This introduces noise — which turns out to be a feature, not a bug, because it helps escape local minima and saddle points.

All variants in this family are essentially answers to the same question: **how can we make the plain gradient step smarter?** The two biggest weaknesses of vanilla SGD are:

1. **It treats all directions equally** — every parameter gets the same learning rate regardless of how "curved" the loss surface is in that direction.
2. **It reacts only to the current gradient** — it has no memory, so it oscillates on ravines and slows down along flat plateaus.

Momentum, Nesterov, and their descendants each attack one or both of these problems.

---

## Vanilla SGD (Stochastic Gradient Descent)

### The Idea

The simplest possible optimizer. On each step, compute the gradient on a mini-batch and subtract it (scaled by the learning rate):

```
θ ← θ - η · ∇L(θ; x_batch, y_batch)
```

### Why It Still Matters

Despite being the oldest optimizer, SGD (with careful tuning) often achieves **better generalization** than adaptive methods like Adam. This is well-documented in computer vision: ResNets, VGGs, and most ImageNet-scale models are trained with SGD + Momentum, not Adam. The intuition is that adaptive methods "overfit" the optimization landscape in a way that hurts test performance.

### The Problem

SGD is extremely sensitive to the learning rate and has no mechanism to accelerate in flat directions or slow down in steep ones. On a loss surface shaped like an elongated valley (which is common in deep networks), it will:

- **Oscillate** wildly across the narrow dimension
- **Creep slowly** along the long dimension toward the minimum

This is exactly what Momentum fixes.

### PyTorch Usage

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,          # Must be tuned carefully
    weight_decay=1e-4 # L2 regularization (optional)
)
```

### When to Use

- **Image classification** with CNNs (especially with a tuned LR schedule)
- When you care more about **final test accuracy** than training speed
- When you have a **well-tuned learning rate schedule** (cosine annealing works great)
- Research papers benchmarking against a strong baseline

---

## SGD with Momentum ⭐ Most Important in This Family

### The Problem It Solves

Imagine rolling a ball down a hilly landscape. Vanilla SGD is like a person taking tiny steps in whatever direction the slope points — they change direction instantly at every step and never build up speed. Momentum is like that ball: it **accumulates velocity** in directions that are consistently downhill and dampens oscillations in noisy directions.

### The Math

Momentum introduces a **velocity vector** `v` that accumulates gradients over time:

```
v_t  = β · v_{t-1} + ∇L(θ_t)
θ_t  = θ_{t-1} - η · v_t
```

Where `β` is the **momentum coefficient** (typically `0.9`), controlling how much of the past velocity is retained.

Expanding this over time:

```
v_t = ∇L_t + β·∇L_{t-1} + β²·∇L_{t-2} + ...
```

This is an **exponentially decaying weighted sum of all past gradients**. Gradients from the distant past contribute less. Recent gradients contribute more. The effect is:

- In directions where gradients consistently point the same way → velocity builds up, effective step size grows → **acceleration**
- In directions where gradients oscillate in sign → they cancel out, velocity stays near zero → **damping of oscillations**

### Intuition: The Ravine Problem

Consider a loss surface that looks like a long, narrow valley:
```
Loss surface cross-section (top view):

     ← gradient direction →
     ↑ narrow dimension (oscillates)
     ↓
     ← long dimension (progress is slow) →
```

With vanilla SGD, the update zigzags across the narrow dimension while barely moving along the long dimension. With momentum, the zigzag components in the narrow dimension cancel out in the velocity accumulator, while the consistent component along the long dimension accumulates → much faster convergence.

### The Effective Learning Rate

A useful way to understand momentum: with `β = 0.9`, the velocity converges to `1/(1-β) = 10` times the gradient magnitude in a constant gradient direction. So momentum effectively **multiplies your learning rate by ~10** in consistent directions, without you having to set a larger `lr` (which would cause instability in other directions).

| β | Effective LR multiplier |
|---|------------------------|
| 0.5 | 2× |
| 0.9 | 10× |
| 0.99 | 100× |

### PyTorch Usage

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,       # Standard value — rarely changed
    weight_decay=1e-4,  # L2 regularization
    nesterov=False       # Set True for Nesterov variant
)
```

### Hyperparameter Guide

| Hyperparameter | Typical Value | Notes |
|----------------|--------------|-------|
| `lr` | 0.1 → 0.01 → 0.001 (scheduled) | Start high, decay with schedule |
| `momentum` | 0.9 | Almost always 0.9; 0.99 for very noisy settings |
| `weight_decay` | 1e-4 to 5e-4 | Standard for image models |

### When to Use

- **The default for CNNs and vision models** trained from scratch
- Whenever you pair it with CosineAnnealingLR or StepLR
- ResNets, EfficientNets, ViTs in supervised image training

### Common Pitfall

Momentum has **inertia** — it cannot stop quickly. If your LR schedule drops the learning rate suddenly, the accumulated velocity can cause the optimizer to "overshoot" the minimum. This is why **warm restarts** (CosineAnnealing with restarts) work well with momentum — the velocity is implicitly reset when the LR jumps back up.

---

## Nesterov Accelerated Gradient (NAG)

### The Intuition

Standard momentum computes the gradient *at the current position*, then takes a step using the accumulated velocity. This is slightly wasteful — you already know you're about to move by roughly `β · v`, so why not **compute the gradient at where you're going to be**, rather than where you are now?

That's exactly what Nesterov does: it's a **look-ahead** correction.

### The Math

```
# Standard Momentum:
v_t = β·v_{t-1} + ∇L(θ_t)
θ_t = θ_{t-1} - η·v_t

# Nesterov:
v_t = β·v_{t-1} + ∇L(θ_t - η·β·v_{t-1})   ← gradient at the "look-ahead" point
θ_t = θ_{t-1} - η·v_t
```

### Does It Matter?

In theory, Nesterov has better convergence bounds for convex problems. In practice for deep learning, the difference over standard momentum is **small but consistently positive** — Nesterov often converges slightly faster and is rarely worse. Most modern training recipes use it.

### PyTorch Usage

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True   # Just flip this flag
)
```

---

## Quick Reference: SGD Family Comparison

| Optimizer | Memory | Adaptive LR? | Typical Use Case |
|-----------|--------|-------------|-----------------|
| SGD | None | No | Baseline, simple problems |
| SGD + Momentum | Velocity | No | CNNs, most vision models |
| Nesterov | Velocity | No | Same as momentum, slight improvement |

---

## Practical Recipe: Training a ResNet with SGD

This is the canonical setup used in most ImageNet papers:

```python
import torch
import torch.nn as nn

model = torchvision.models.resnet50()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,           # Start high
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# Decay LR by 10x at epochs 30, 60, 90 (for 100 epoch training)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)

for epoch in range(100):
    train_one_epoch(model, optimizer)
    scheduler.step()
```

This setup achieves ~76% top-1 accuracy on ImageNet with ResNet-50.

---

## Key Takeaways

- **Vanilla SGD** is the foundation — understand it before anything else.
- **SGD + Momentum (β=0.9)** is the workhorse of computer vision. When paired with a good LR schedule, it often outperforms adaptive methods on final test accuracy.
- **Nesterov** is a free upgrade — just set `nesterov=True`, it's almost always at least as good.
- The SGD family requires more **learning rate tuning** than adaptive methods but rewards that effort with better generalization.
- For transformers and NLP tasks, Adam/AdamW tend to dominate — SGD struggles without careful tuning on those architectures.