# Modern Optimizers — Lion, Adafactor, LAMB

---

## What Are "Modern" Optimizers?

Adam and SGD+Momentum were designed in an era where models had millions of parameters. The modern landscape looks very different: billion-parameter language models, training runs that cost millions of dollars, models deployed on edge devices with memory constraints.

The optimizers in this category were invented to address problems that only become severe at scale:

1. **Memory overhead**: Adam stores two copies of every parameter (the first and second moment estimates `m` and `v`). For a 70B parameter model, that's an extra ~560GB of optimizer state in float32. This is often the bottleneck that prevents training larger models.
2. **Communication cost**: In distributed training across hundreds of GPUs, optimizer state must be synchronized. Smaller optimizer state = less communication overhead.
3. **Stability at scale**: Very large models and very large batch sizes expose numerical stability issues that don't appear in smaller experiments.

Modern optimizers are generally about picking one or more of these tradeoffs differently than Adam.

---

## Lion — Evolved Optimizer ⭐ Most Interesting

### The Origin

Lion (EvoLved Sign Momentum, Chen et al., Google Brain 2023) wasn't designed by human intuition — it was **discovered by a program search algorithm** that searched the space of possible optimizer update rules. The researchers defined a "meta-optimizer" that combined basic mathematical operations and evolved candidate update rules, evaluated them on proxy tasks, and kept the best-performing ones. Lion emerged from this search.

The fact that a machine-discovered optimizer outperforms human-designed ones is itself a notable result.

### The Math

Lion's update rule is remarkably simple — simpler than Adam:

```
# Adam update (for reference):
m_t = β₁·m_{t-1} + (1-β₁)·g_t
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)   # Two moment estimates, division

# Lion update:
c_t = β₁·m_{t-1} + (1-β₁)·g_t            # Interpolation (no bias correction needed)
θ_t = θ_{t-1} - η · sign(c_t)             # Just the SIGN — no division
m_t = β₂·m_{t-1} + (1-β₂)·g_t            # Update momentum AFTER the step
```

The key operation is `sign(c_t)`: every parameter update is exactly `+η` or `-η`. No more, no less.

### Why Sign Updates?

This seems radical — why discard the gradient magnitude entirely?

**The case for it:**

1. **All parameters move the same distance per step.** This is the ultimate form of adaptive learning rate — the gradient magnitude is used only to determine *direction*, not *step size*. Parameters that receive large gradients don't automatically get to move further than parameters with small gradients.

2. **Memory efficiency.** Lion only maintains one moment vector `m` instead of Adam's two (`m` and `v`). For a 1B parameter model in float32, this saves ~4GB of optimizer state.

3. **Implicit regularization.** Sign updates have a regularizing effect similar to L∞ normalization — they prevent any single parameter from dominating the update.

4. **It actually works.** In the original paper, Lion matched or outperformed Adam/AdamW on ImageNet, BERT fine-tuning, diffusion models, and code generation benchmarks while using less memory and often requiring fewer training steps.

### Hyperparameter Differences from Adam

Because every update is `±η` (a fixed magnitude), the learning rate means something different than in Adam:

- **Lion needs a smaller learning rate** than AdamW — typically `3-10x smaller`
- The original paper suggests `lr ≈ 1e-4` where AdamW would use `lr ≈ 1e-3`

```python
# Using Lion with PyTorch (install lion-pytorch)
# pip install lion-pytorch
from lion_pytorch import Lion

optimizer = Lion(
    model.parameters(),
    lr=1e-4,           # ~3-10x smaller than AdamW equivalent
    betas=(0.9, 0.99), # Default betas, β₂ closer to β₁ than Adam
    weight_decay=1e-2  # Still supports decoupled weight decay
)
```

### When to Use Lion

- Training large vision models, diffusion models (SDXL-style)
- When optimizer memory is a bottleneck (training larger models in same VRAM)
- Fine-tuning large language models
- When you want to experiment with something genuinely different from the Adam family

### Caveats

Lion is newer and less battle-tested than AdamW. Some practitioners report instability in very large LLM training runs. It's a promising direction but not yet the universal replacement for AdamW that some early papers suggested.

---

## Adafactor

### The Problem It Solves

Adam requires storing two full copies of all parameters (moments `m` and `v`). Adafactor (Shazeer & Stern, 2018) was designed specifically for **extreme memory efficiency** in large models — particularly the large embedding matrices in sequence-to-sequence models.

The key insight: the second moment matrix `v` has a lot of structure. For a weight matrix `W ∈ ℝ^{m×n}`, instead of storing a full `m×n` second moment matrix, Adafactor factors it into a row vector `R ∈ ℝ^m` and a column vector `C ∈ ℝ^n`, approximating:

```
v ≈ R · Cᵀ / (1ᵀC)
```

This reduces memory from `O(m·n)` to `O(m+n)` — for a 32k×512 embedding matrix, that's from 16M to 33k stored values. A ~500x reduction.

Additionally, Adafactor removes the first moment estimate `m` by default (no momentum), using a relative learning rate schedule instead of a fixed `lr`.

### In Practice

Adafactor was used to train **T5** and is still used in some Google models. It's particularly relevant when training models with very large vocabulary sizes or embedding dimensions where memory is the hard constraint.

```python
from transformers import Adafactor

optimizer = Adafactor(
    model.parameters(),
    relative_step=True,     # Auto-compute LR based on step count
    warmup_init=True,       # Warmup from small LR
    scale_parameter=True
)
# Note: when relative_step=True, don't pair with an external LR scheduler
```

### When to Use Adafactor

- Training very large models with huge embedding matrices (T5-scale, seq2seq)
- When you're extremely memory-constrained
- As an alternative when reproducing Google's T5/Flan results

For most practitioners using modern hardware with reasonable VRAM, AdamW is simpler and more predictable.

---

## LAMB

### The Problem It Solves

LAMB (Layer-wise Adaptive Moments optimizer for Batch training, You et al., 2019) was designed specifically for **very large batch training** — specifically BERT training with batch sizes of 32,768 or larger.

When you scale batch size, you typically need to scale the learning rate proportionally (linear scaling rule). But with Adam, this breaks down at very large batch sizes because different layers have different gradient norms and the same LR doesn't work well for all of them.

LAMB adds a **per-layer trust ratio** that normalizes the Adam update by the ratio of the parameter norm to the update norm:

```
# Adam update direction:
u_t = m̂_t / (√v̂_t + ε) + λ·θ_t

# LAMB step:
trust_ratio = ||θ_t|| / ||u_t||    # Layer-wise scaling
θ_t = θ_{t-1} - η · trust_ratio · u_t
```

This ensures that no layer's parameters move more than a fixed fraction of their current magnitude in a single step — preventing runaway updates in some layers while allowing larger updates in others.

### Result

LAMB allowed BERT pretraining in **76 minutes** on 1,024 TPUs (batch size 32,768), compared to the original 3+ days. It makes distributed training at extreme scale feasible.

### When to Use LAMB

- **Distributed training with very large batch sizes** (32k+)
- Replicating BERT-style pretraining at scale
- Not needed for single-GPU or small-scale training — LAMB provides no benefit at normal batch sizes

---

## Summary Comparison

| Optimizer | Memory Overhead | Best For | Key Innovation |
|-----------|----------------|----------|----------------|
| Adam/AdamW | 2× params | General deep learning | Adaptive LR + momentum |
| Lion | 1× params | Large models, diffusion | Sign updates, memory efficient |
| Adafactor | ~0× params | Huge embeddings, T5-scale | Factored second moments |
| LAMB | 2× params | Large batch distributed training | Layer-wise trust ratios |

---

## Key Takeaways

- **Lion** is the most practically interesting modern optimizer — genuinely different from the Adam family, backed by surprising research origins, and showing real advantages in vision and generation tasks.
- **Adafactor** is the optimizer to reach for when memory is the hard constraint and you're working with T5-scale models.
- **LAMB** is highly specialized — it solves a problem (ultra-large batch training) that most practitioners never encounter.
- None of these have fully displaced AdamW as the default. AdamW remains the safe, well-understood choice. These are worth knowing and trying when you hit the specific limitations they address.