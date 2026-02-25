# Adaptive Optimizers — Adam, AdamW, RMSProp, AdaGrad

---

## What Are Adaptive Optimizers?

The core idea behind the SGD family is simple: every parameter gets the same learning rate. You decide on a global `lr`, and every weight in your network — from the first convolutional filter to the last classification layer — is updated with that same step size.

This is often a bad idea.

In a deep network, different parameters have wildly different gradient scales. Embedding layers for rare words receive sparse, occasionally large gradients. BatchNorm scales receive dense, small gradients. Applying a single learning rate to both means you're either:
- **Too aggressive** for some parameters (causing instability), or
- **Too conservative** for others (causing slow learning)

**Adaptive optimizers solve this by giving each parameter its own effective learning rate**, automatically inferred from the history of that parameter's gradients.

The general principle:

```
effective_lr(parameter_i) = global_lr / (some_measure_of_past_gradient_magnitude_i)
```

Parameters that have been receiving large gradients → small effective LR (slow down, be careful)
Parameters that have been receiving small gradients → large effective LR (speed up, keep moving)

This makes adaptive methods:
- **Much less sensitive** to the global learning rate choice
- **Faster to converge** in early training
- **Self-tuning** across layers with very different gradient scales
- The **default choice for NLP, transformers, and most modern architectures**

The tradeoff: they can **generalize slightly worse** than well-tuned SGD+Momentum on some tasks (notably image classification), because the adaptive scaling can reduce useful gradient signal.

---

## AdaGrad — The Origin

AdaGrad (Adaptive Gradient, 2011) was the first adaptive optimizer. It accumulates the **sum of squared gradients** for each parameter and divides the update by the square root of this sum:

```
G_t = G_{t-1} + (∇L_t)²          # Accumulate squared gradients
θ_t = θ_{t-1} - (η / √(G_t + ε)) · ∇L_t
```

**The insight:** Parameters that received large gradients in the past get a smaller effective learning rate. Parameters that received small or sparse gradients (like rare-word embeddings) get a larger effective learning rate.

**The fatal flaw:** `G_t` only ever grows — it accumulates forever. Eventually, every effective learning rate shrinks to near zero and learning stops. This makes AdaGrad unusable for non-convex deep learning where you train for many epochs.

**Where it's still used:** Convex problems, sparse linear models, some NLP feature-based methods. Not used directly in deep learning anymore, but it's the intellectual parent of RMSProp and Adam.

---

## RMSProp

RMSProp (Root Mean Square Propagation, Hinton 2012 — never formally published, introduced in a Coursera lecture) fixed AdaGrad's "learning rate dies" problem by replacing the cumulative sum with an **exponentially weighted moving average** of squared gradients:

```
v_t = β · v_{t-1} + (1-β) · (∇L_t)²     # Decaying average, not cumulative sum
θ_t = θ_{t-1} - (η / √(v_t + ε)) · ∇L_t
```

Now `v_t` forgets old gradients — the denominator reflects **recent** gradient magnitude, not all-time history. This means the effective learning rate can recover if gradients suddenly become smaller.

**Typical values:** `β = 0.9`, `ε = 1e-8`, `lr = 0.001`

**Where it's used:** RNNs (it was the go-to for RNNs before transformers), reinforcement learning (still widely used), and as the "momentum component" inside Adam.

RMSProp essentially became a stepping stone — Adam combined it with momentum and added bias correction, making RMSProp largely obsolete for most tasks. It's still used in RL (PPO, A3C) where Adam's bias correction isn't needed and simpler updates are preferred.

---

## Adam — The Default Optimizer ⭐ Most Important

### The Idea

Adam (Adaptive Moment Estimation, Kingma & Ba, 2014) is the most widely used optimizer in deep learning. It combines two ideas:

1. **Momentum** (first moment): a moving average of the gradients themselves
2. **RMSProp** (second moment): a moving average of the squared gradients

And adds a crucial **bias correction** step that makes the first few updates well-scaled.

### The Math

```
# Compute gradients
g_t = ∇L(θ_t)

# Update first moment (momentum — direction)
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t

# Update second moment (RMSProp — magnitude)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

# Bias correction (critical for early steps)
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

# Parameter update
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

### Why Bias Correction Matters

At step `t=1`, both `m_0` and `v_0` are initialized to zero. Without correction:

- `m_1 = (1 - β₁) · g_1 = 0.1 · g_1` — this massively underestimates the true gradient
- `v_1 = (1 - β₂) · g_1² = 0.001 · g_1²` — even more underestimated

Without bias correction, early Adam updates are tiny and nearly useless. The correction `m̂_t = m_t / (1 - β₁ᵗ)` rescales these early estimates to be properly calibrated. After many steps, `(1 - β₁ᵗ) → 1` and the correction vanishes — it's only important early in training.

### Intuition: What Adam Is Really Doing

The update `m̂_t / √v̂_t` can be read as:

> "Move in the average gradient direction, but scale the step inversely by how variable this parameter's gradients have been."

If a parameter's gradient is consistently `+5`, then `m̂ ≈ 5` and `√v̂ ≈ 5`, so the step is `≈ η`. Large but consistent gradient → moderate step.

If a parameter's gradient randomly oscillates between `+5` and `-5`, then `m̂ ≈ 0` and `√v̂ ≈ 5`, so the step is `≈ 0`. Noisy direction → nearly no step.

If a parameter rarely receives gradient but when it does it's `0.001`, then `m̂ ≈ 0.001` and `√v̂ ≈ 0.001`, so the step is `≈ η`. Tiny but consistent gradient → full step (great for sparse features, rare words).

### Default Hyperparameters

| Hyperparameter | Default Value | Notes |
|----------------|--------------|-------|
| `lr` | 1e-3 | Often reduced to 1e-4 for fine-tuning |
| `β₁` | 0.9 | Momentum coefficient — rarely changed |
| `β₂` | 0.999 | Variance coefficient — rarely changed |
| `ε` | 1e-8 | Numerical stability — sometimes set to 1e-6 or 0.1 for transformers |

The defaults are remarkably robust. Most practitioners only tune `lr`.

### PyTorch Usage

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0    # ⚠️ Don't use this — use AdamW instead
)
```

### Adam's Problem: L2 Regularization Is Broken

When you add `weight_decay` to Adam, it adds `λ·θ` to the gradient before the adaptive scaling:

```
effective_decay = λ·θ / √v̂   ← scaled by gradient variance!
```

This means the weight decay is **different for every parameter** depending on its gradient history. Parameters with large gradient variance (high `√v̂`) get *less* regularization than parameters with small gradient variance. This is mathematically wrong — it means Adam's "weight decay" is not L2 regularization at all.

This is why **AdamW was invented**.

### When to Use Adam

- Quick experimentation and prototyping
- When you don't need regularization (e.g., early exploration)
- Technically, you should almost always prefer AdamW — but Adam remains in widespread use

---

## AdamW — Adam with Fixed Weight Decay ⭐ The Modern Standard

### The Fix

AdamW (Loshchilov & Hutter, 2017) is a minimal modification of Adam: instead of adding weight decay to the gradient (which then gets scaled adaptively), it **applies weight decay directly to the parameters**, bypassing the adaptive scaling:

```
# Adam (broken):
g_t = ∇L(θ_t) + λ·θ_t      # λ·θ added to gradient → gets scaled by 1/√v̂
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)

# AdamW (correct):
g_t = ∇L(θ_t)               # Pure gradient
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε) - η · λ · θ_{t-1}   # Decay applied directly
```

This seems like a small change, but it has significant practical impact:

- Weight decay is now **uniform across all parameters** regardless of gradient variance
- Training stability is improved, especially in later stages of training
- It's the **standard for transformer training** (BERT, GPT, and all descendants use AdamW)

### Typical Hyperparameters for Transformers

| Setting | Value | Notes |
|---------|-------|-------|
| `lr` | 1e-4 to 5e-5 | For fine-tuning; 1e-3 to 3e-4 for training from scratch |
| `betas` | (0.9, 0.999) | Standard; some use (0.9, 0.95) for LLMs |
| `weight_decay` | 0.01 to 0.1 | 0.01 is the BERT default; 0.1 for GPT-style models |
| `eps` | 1e-8 or 1e-6 | 1e-6 slightly more stable for mixed precision |

### PyTorch Usage

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

### Parameter Groups — Don't Decay Everything

A standard practice in transformer training is to **not apply weight decay to biases and LayerNorm parameters**:

```python
# Separate parameters into two groups
no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() 
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters() 
                   if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0    # No decay for biases and norms
    }
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
```

This is used in almost all serious transformer training code (HuggingFace Transformers, nanoGPT, etc.).

### AdamW vs Adam: Which to Use?

**Use AdamW by default.** It strictly improves upon Adam for any task with regularization. The only reason to use vanilla Adam is legacy code or when explicitly not using weight decay.

---

## Comparison Table

| Optimizer | Adaptive LR | Momentum | Weight Decay | Best For |
|-----------|------------|----------|--------------|----------|
| AdaGrad | Yes | No | No | Convex / sparse (historical) |
| RMSProp | Yes | No | No | RNNs, Reinforcement Learning |
| Adam | Yes | Yes | Broken (L2) | Quick experiments |
| AdamW | Yes | Yes | Correct (decoupled) | **Transformers, NLP, fine-tuning — default** |

---

## Practical Recipe: Fine-Tuning a Transformer with AdamW

```python
from transformers import AdamW, get_linear_schedule_with_warmup

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Parameter groups
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)

# Warmup + linear decay schedule (standard for BERT fine-tuning)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps
)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

---

## Key Takeaways

- **AdaGrad** → historical significance, not used in deep learning directly anymore.
- **RMSProp** → still relevant for reinforcement learning; building block for Adam.
- **Adam** → excellent defaults, widely used, but use AdamW if you need regularization.
- **AdamW** → the modern standard for transformers and NLP. Should be your default for any attention-based model.
- All adaptive methods are more **forgiving of LR choice** than SGD, but AdamW + LR warmup + cosine decay is the combination that dominates modern practice.