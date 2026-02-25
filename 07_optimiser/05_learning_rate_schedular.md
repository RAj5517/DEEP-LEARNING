# Learning Rate Schedulers

---

## What Are Learning Rate Schedulers?

The learning rate is the single most important hyperparameter in training a neural network. But treating it as a constant throughout training is leaving performance on the table.

The intuition for why learning rates should change over time comes from thinking about the loss landscape:

**Early in training:** The parameters are randomly initialized and far from any good solution. Large steps are fine — you want to move quickly across the landscape, not get stuck near initialization.

**Late in training:** You're near a good solution. Large steps will cause you to overshoot the minimum and bounce around. Small, precise steps are needed to settle into a low-loss region.

**The conclusion:** The optimal learning rate is not constant — it should be **large early and small late**. This is what learning rate schedulers automate.

But it's more nuanced than simply "start big, end small":

- **Too fast decay**: You shrink the LR before the model has found a good region → converge to a suboptimal solution
- **Too slow decay**: You never settle into the minimum → training stays noisy and doesn't fully converge
- **Sharp drops** (like StepLR) can cause instability if timed poorly
- **Smooth decay** (like cosine) tends to be more stable and often achieves better final accuracy

Schedulers are not optional accessories — they are a **critical part of the training recipe**. The same optimizer with different schedulers can produce wildly different results.

---

## StepLR — Simple Decay

### The Idea

The most straightforward scheduler: multiply the learning rate by a fixed factor `γ` every `step_size` epochs.

```
LR at epoch t:  η · γ^(floor(t / step_size))
```

If you start with `lr=0.1`, `step_size=30`, `γ=0.1`:
- Epochs 0–29: LR = 0.1
- Epochs 30–59: LR = 0.01
- Epochs 60–89: LR = 0.001
- Epochs 90+: LR = 0.0001

### PyTorch Usage

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,   # Decay every 30 epochs
    gamma=0.1       # Multiply by 0.1 at each step
)

# In training loop:
for epoch in range(90):
    train_one_epoch(...)
    scheduler.step()
```

### MultiStepLR (Better Version)

Instead of regular intervals, specify exact epochs for decay:

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 80],  # Decay at these specific epochs
    gamma=0.1
)
```

### When to Use

StepLR is simple, interpretable, and works reliably for CNNs with known training durations. It's the scheduler in the original ResNet paper (decay at epochs 30/60/90 for 100-epoch training).

**Weakness:** The sharp drops can cause training instability, and you need to manually tune the milestones for each training duration. CosineAnnealingLR is generally preferred today.

---

## CosineAnnealingLR — The Modern Standard 🔥

### The Idea

Instead of discrete drops, smoothly decay the learning rate following a **half-cosine curve** from `η_max` down to `η_min`:

```
η_t = η_min + (1/2)(η_max - η_min) · (1 + cos(π · t/T))
```

Where `t` is the current step and `T` is the total number of steps.

At `t=0`: `cos(0) = 1` → LR = `η_max`
At `t=T/2`: `cos(π/2) = 0` → LR = `(η_max + η_min) / 2`
At `t=T`: `cos(π) = -1` → LR = `η_min`

The cosine shape starts decaying slowly, accelerates decay in the middle, then slows again near the end. This **mirrors the geometry of training**: early training makes rapid progress (large LR okay), middle training refines (smoothly decreasing LR), late training needs precision (small LR).

### PyTorch Usage

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,  # Number of epochs for one cosine cycle
    eta_min=1e-6       # Minimum LR (default 0, but 1e-6 is better)
)
```

### The Full Training Loop

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

for epoch in range(200):
    train_one_epoch(model, optimizer, train_loader)
    val_loss = validate(model, val_loader)
    scheduler.step()
```

### CosineAnnealingWarmRestarts (SGDR)

A popular extension: after reaching `η_min`, **restart** back to `η_max` and repeat. This prevents the model from getting trapped in sharp minima by periodically allowing it to escape with a large learning rate:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,     # Length of first cycle (epochs)
    T_mult=2,   # Each subsequent cycle is T_mult longer (50 → 100 → 200)
    eta_min=1e-6
)
```

The restart trick: when the LR jumps up, the optimizer escapes local minima it was settling into. When it anneals down again, it settles into potentially better minima. This "simulated annealing" behavior often finds flatter, more generalizable minima.

### Why Cosine Is the Default Today

- **Smooth decay** avoids the destabilizing step-changes of StepLR
- **Works regardless of total training duration** — just set `T_max` to the total epochs
- **Consistently achieves better final accuracy** than StepLR across vision and NLP tasks
- Pairs well with both SGD and AdamW
- The warm restart variant helps escape local minima

**Rule of thumb:** When in doubt about which scheduler to use, use CosineAnnealingLR.

---

## OneCycleLR — Superconvergence 🔥

### The Idea

OneCycleLR implements the **1-Cycle Policy** (Smith & Topin, 2018), which discovered something counterintuitive: you can train for **much fewer epochs** if you allow the learning rate to **increase first** before decreasing.

The policy has three phases:
1. **Warmup phase** (~30% of training): LR increases from `lr/div_factor` to `max_lr`
2. **Annealing phase** (~70% of training): LR decreases from `max_lr` to `min_lr` (using cosine)
3. **Optional final phase**: Anneal to very small LR

Simultaneously, momentum follows the **inverse** curve: high when LR is small, low when LR is large. This is because high momentum + high LR causes instability, so they trade off.

### The Math Intuition

Why does increasing the LR first help? The hypothesis is that the high-LR phase acts as a **regularizer** — the large steps prevent the model from settling into sharp, overfit minima. The model is forced to find flat, generalizable regions of the loss surface that are stable under large perturbations. Then the annealing phase polishes the solution.

### PyTorch Usage

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,                    # Peak learning rate
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.3,                 # 30% of training for warmup phase
    div_factor=25,                 # Initial LR = max_lr / 25 = 0.004
    final_div_factor=1e4,          # Final LR = max_lr / (25 * 1e4) ≈ tiny
    anneal_strategy='cos'          # Cosine annealing
)

# IMPORTANT: Call scheduler.step() every BATCH, not every epoch
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # ← After every batch!
```

### Finding max_lr: The LR Range Test

A practical utility from the same paper: run a short training loop where you increase the LR exponentially from very small to very large, and plot the loss. The ideal `max_lr` is just before the loss starts to diverge:

```python
from torch_lr_finder import LRFinder

optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot()  # Look for the point of steepest descent
```

### OneCycleLR vs CosineAnnealingLR

| | CosineAnnealingLR | OneCycleLR |
|--|-------------------|-----------|
| Training duration | Any | Short to medium |
| Convergence speed | Moderate | Fast (superconvergence) |
| Final accuracy | Often higher | Competitive |
| Tuning difficulty | Low | Medium (need to find max_lr) |
| Best with | SGD, AdamW | SGD especially |
| Called per | Epoch | Batch |

**When to choose OneCycleLR:** When you want fast convergence and are willing to do a brief LR range test. Particularly effective with SGD for image classification. Also used with AdamW in some transfer learning setups.

**When to choose CosineAnnealingLR:** When you want a reliable, well-tested default for longer training runs. Less tuning required.

---

## ReduceLROnPlateau — Adaptive to Validation

### The Idea

All previous schedulers follow a fixed schedule determined before training starts. ReduceLROnPlateau takes a different approach: **monitor validation performance** and reduce the LR when improvement stalls.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # 'min' for loss, 'max' for accuracy
    factor=0.5,       # Multiply LR by this when triggered
    patience=10,      # Wait this many epochs with no improvement before decaying
    min_lr=1e-6,      # Don't go below this
    verbose=True      # Print when LR changes
)

# In training loop:
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)   # ← Pass the metric to monitor
```

### When to Use

- When you don't know in advance how long to train or at what points to decay
- When you want the scheduler to respond to actual training dynamics rather than a fixed plan
- Useful for fine-tuning on small datasets where training duration is unpredictable
- **Not ideal for transformers** where warmup is needed and validation metrics can be noisy

### Limitations

- Reactive rather than proactive — it can reduce LR too late
- The `patience` parameter needs tuning based on how noisy your validation metric is
- Can be unpredictable in behavior across runs if validation is noisy

---

## Warmup Variants ⭐ Essential for Transformers

### Why Transformers Need Warmup

Transformers with attention mechanisms have a specific problem in early training: the attention weights are essentially random, producing noisy gradient signals. If you start with the full learning rate immediately:

1. The Adam second moment estimate `v` is initialized to zero and takes many steps to warm up — early adaptive scaling is unreliable
2. Large random gradients from noisy attention can cause immediate parameter explosion
3. The model can end up in a bad region of the loss landscape it can never escape

The solution: **start with a tiny learning rate and increase it gradually** to give the optimizer time to accumulate reliable moment estimates and give the model time to find a stable initial direction.

### Linear Warmup

The simplest warmup: linearly increase from 0 (or a small value) to the target LR over the first N steps:

```python
from transformers import get_linear_schedule_with_warmup

total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

This was used in the original BERT paper and remains a common choice for fine-tuning.

### Cosine with Warmup (The Modern Standard for Transformers)

The dominant scheduler for transformer pretraining: linear warmup followed by cosine decay. This combines the stability benefits of warmup with the smooth decay of cosine annealing.

```
Phase 1 (steps 0 → warmup_steps):    LR linearly increases from ~0 to max_lr
Phase 2 (steps warmup_steps → total): LR follows cosine curve from max_lr to min_lr
```

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    num_cycles=0.5   # Half cosine (default) — doesn't go back up
)
```

This is the scheduler used in GPT-2, GPT-3, BERT, T5, and essentially all modern large language models.

### How Much Warmup?

| Setting | Typical warmup |
|---------|---------------|
| BERT fine-tuning | 6-10% of total steps |
| BERT pretraining | 10,000 steps (out of ~1M) |
| GPT-style pretraining | 2,000-4,000 steps |
| ViT from scratch | 10,000 steps |

The required warmup duration scales roughly with model size — larger models need longer warmup.

### Constant LR with Warmup

Some practitioners find that just doing warmup + constant LR outperforms warmup + decay for fine-tuning tasks. Worth experimenting with:

```python
from transformers import get_constant_schedule_with_warmup

scheduler = get_constant_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps
)
```

### Implementing Warmup Without External Libraries

```python
import math

def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## Scheduler Decision Guide

```
Which scheduler should I use?

Are you training a Transformer (BERT, GPT, ViT, etc.)?
  └─ YES → Cosine with Linear Warmup (use get_cosine_schedule_with_warmup)

Are you training a CNN for image classification?
  └─ Short training (<50 epochs) or want fast convergence?
       └─ OneCycleLR (run LR range test first)
  └─ Standard training (>50 epochs)?
       └─ CosineAnnealingLR (T_max = total epochs, eta_min = 1e-6)

Do you not know how long to train or when validation will plateau?
  └─ ReduceLROnPlateau (patience = 5-10 epochs)

Are you reproducing a classic paper (ResNet, VGG, etc.)?
  └─ MultiStepLR with the paper's milestones
```

---

## Complete Reference: All Schedulers

| Scheduler | Schedule Shape | Called Per | Notes |
|-----------|---------------|-----------|-------|
| StepLR | Staircase | Epoch | Simple, interpretable |
| MultiStepLR | Staircase (irregular) | Epoch | Classic for CNNs |
| CosineAnnealingLR | Smooth S-curve | Epoch | Modern default for vision |
| CosineAnnealingWarmRestarts | Cosine with resets | Epoch or Step | Helps escape local minima |
| OneCycleLR | Up then down | **Batch** | Fast training, needs max_lr tuning |
| ReduceLROnPlateau | Reactive | Epoch | When training duration unknown |
| Linear Warmup + Cosine | Ramp up then cosine | **Step** | Standard for transformers |
| Linear Warmup + Linear Decay | Ramp up then linear | **Step** | Classic for BERT fine-tuning |

---

## Key Takeaways

- **Always use a scheduler.** Constant LR almost always leaves accuracy on the table.
- **CosineAnnealingLR** is the safe default for vision/CNN tasks — smooth, reliable, no milestone tuning.
- **OneCycleLR** is the choice for faster convergence — particularly powerful with SGD; requires finding `max_lr`.
- **Warmup + Cosine decay** is the standard for all transformer-based models — non-negotiable for large models.
- **ReduceLROnPlateau** is best for fine-tuning on unknown-length tasks or when you want adaptive behavior.
- OneCycleLR is called per **batch**, others are called per **epoch** — mixing this up is a very common bug.