# Hyperparameter Tuning — Practical Guide

---

## The Priority Order

Not all hyperparameters are equal. Tune in this order — the earlier ones matter more:

```
1. Learning Rate          ← biggest impact by far
2. Batch Size             ← tightly coupled with LR
3. Weight Decay           ← second most impactful regularizer
4. Warmup Steps           ← non-negotiable for transformers
5. Scheduler Choice       ← defines the LR trajectory
6. Gradient Clip Value    ← stability, not performance
7. Dropout Rate           ← architecture-level regularization
8. Number of Epochs       ← when to stop
9. Optimizer Choice       ← usually decided upfront
10. Architecture Depth    ← rarely tuned mid-project
```

---

## Learning Rate ⭐⭐⭐

The single most important hyperparameter. Everything else is secondary.

**Starting points by optimizer:**

| Optimizer | Typical Range | Common Default |
|-----------|-------------|----------------|
| SGD + Momentum | 0.001 – 0.1 | 0.01 or 0.1 |
| Adam | 1e-4 – 1e-3 | 1e-3 |
| AdamW (fine-tuning) | 1e-5 – 5e-4 | 2e-5 – 5e-5 |
| AdamW (from scratch) | 1e-4 – 3e-4 | 1e-3 |

**The LR Range Test** — the fastest way to find a good LR: increase LR exponentially from 1e-7 to 10 over ~100 batches, plot loss vs LR, pick the LR just before loss stops decreasing:

```python
from torch_lr_finder import LRFinder

optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot()
# Use ~1/3 to 1/10 of the LR at minimum loss as your starting LR
```

**Signs you have the wrong LR:**

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Loss explodes immediately | LR too high | Divide by 10 |
| Loss barely moves | LR too low | Multiply by 10 |
| Loss decreases then plateaus early | LR too high (later) | Use a schedule |
| Loss oscillates but trends down | Slightly high | Mild reduction or add schedule |

**LR is always set relative to batch size** — if you double batch size, you generally need to scale LR (see Batch Size section).

---

## Batch Size

Batch size and LR are coupled — changing one often requires changing the other.

**Practical ranges:**

| Hardware | Typical Batch Size |
|----------|-------------------|
| Single GPU (16GB) | 32 – 256 |
| Single GPU (40GB+) | 64 – 512 |
| Multi-GPU | Scale up with gradient accumulation |

**Linear scaling rule:** When you multiply batch size by `k`, multiply LR by `k`.

```
batch=32,  lr=0.01  →  batch=128, lr=0.04   (4× batch → 4× lr)
```

This rule holds reasonably well up to large batch sizes but breaks down above ~4k–8k. For very large batches, use LAMB or warmup to stabilize.

**Gradient Accumulation** — simulate large batches on limited memory:

```python
accumulation_steps = 4   # Effective batch = actual_batch × 4

optimizer.zero_grad()
for i, (images, labels) in enumerate(train_loader):
    loss = model(images, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Small batch sizes (≤16)** act as implicit regularization due to gradient noise — useful when overfitting on small datasets. Large batches converge faster per epoch but can generalize worse without compensating regularization.

---

## Weight Decay Strength

Second most impactful hyperparameter after LR. Acts as the main capacity control alongside dropout.

**Quick reference:**

| Setting | Value |
|---------|-------|
| SGD + CNN (ImageNet) | 1e-4 |
| AdamW + Transformer (fine-tuning) | 0.01 |
| AdamW + Transformer (pretraining) | 0.1 |
| AdamW + ViT | 0.05 – 0.3 |

**Rule of thumb:** Larger dataset → smaller weight decay. Smaller dataset → larger weight decay.

If you're overfitting: increase weight decay before adding dropout. If you're underfitting: reduce weight decay first.

Always use **decoupled weight decay** (AdamW, not Adam with weight_decay). Always exclude biases and norm layers from weight decay:

```python
no_decay = ['bias', 'LayerNorm.weight']
params = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
optimizer = torch.optim.AdamW(params, lr=1e-4)
```

---

## Warmup Steps ⭐ (Transformers)

**Non-negotiable for transformers.** Without warmup, the Adam second moment `v` starts at zero and produces unreliable adaptive scaling — early updates can be destructive.

**How much warmup:**

| Task | Warmup |
|------|--------|
| BERT fine-tuning | 6–10% of total steps |
| BERT pretraining | ~10,000 steps |
| GPT pretraining | 2,000 – 4,000 steps |
| ViT from scratch | 10,000 steps |
| Small model / short training | 500 – 1,000 steps |

```python
from transformers import get_cosine_schedule_with_warmup

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.06 * total_steps)   # 6% warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

For CNNs, warmup is optional but can help stability at high LR. Start without it; add if training is unstable in early epochs.

---

## Scheduler Choice

Your scheduler defines the entire LR trajectory. Choose based on what you're training:

| Scheduler | When to Use | Key Setting |
|-----------|------------|-------------|
| **CosineAnnealingLR** | CNNs, standard training | `T_max=num_epochs`, `eta_min=1e-6` |
| **OneCycleLR** | Fast training, SGD | `max_lr` from LR range test |
| **Cosine + Warmup** | Any transformer | `num_warmup_steps=6–10%` |
| **ReduceLROnPlateau** | Unknown training duration, fine-tuning | `patience=5–10` |
| **StepLR / MultiStepLR** | Replicating classic papers | Match paper milestones |

Decision shortcut:
- Transformer → **Cosine + Warmup**
- CNN, fixed epochs → **CosineAnnealingLR**
- CNN, fast convergence → **OneCycleLR**
- Fine-tuning, unknown duration → **ReduceLROnPlateau**

---

## Gradient Clip Value

Gradient clipping prevents training instability caused by exploding gradients — it caps the global gradient norm at a threshold. This is about **stability**, not performance.

```python
# Apply before optimizer.step(), after loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Standard values:**

| Model Type | Clip Value |
|-----------|-----------|
| Transformers (LLMs, BERT) | 1.0 |
| RNNs / LSTMs | 1.0 – 5.0 |
| CNNs | Not usually needed |
| Diffusion models | 1.0 |

**When you need it:** If loss spikes suddenly during training (NaN or large jump), gradient clipping is the fix. For transformers, always include it — it's a cheap safety net. For CNNs with BatchNorm, it's rarely necessary.

Only tune this value if you're seeing persistent instability. `1.0` works for almost all transformer training.

---

## Dropout Rate

Covered in detail in the Dropout file. Quick reference for tuning:

| Architecture | Typical Rate |
|-------------|-------------|
| FC layers | 0.3 – 0.5 |
| Transformer (attention/FFN) | 0.1 |
| CNN feature layers | 0.1 – 0.2 (spatial dropout) |
| DropPath (ViT/Swin) | 0.1 – 0.3 (scaled with depth) |

**Tune this last.** Fix your LR, weight decay, and schedule first. Then if you're still overfitting, increase dropout. If underfitting, decrease it.

---

## Number of Epochs

There's no universal answer — it depends on dataset size, model size, and LR schedule. The practical approach:

**For fixed-schedule training (cosine, step):** Set epochs, watch the validation curve. If val loss is still decreasing at the end, train longer. If val loss plateaued 30% through, reduce epochs.

**For adaptive training:** Use early stopping with patience=10–20 and let the data decide.

**Common reference points:**

| Task | Typical Epochs |
|------|---------------|
| ImageNet (ResNet from scratch) | 90 – 300 |
| CIFAR-10/100 (from scratch) | 200 |
| BERT fine-tuning | 3 – 5 |
| ViT fine-tuning | 10 – 30 |
| Small dataset fine-tuning | 5 – 20 |

The right number is: enough for the LR schedule to complete and the model to converge, but not so many that overfitting sets in. With cosine scheduling, train until the LR reaches `eta_min` and a bit more. With early stopping, set a high max and let it run.

---

## Optimizer Choice

This is usually decided upfront based on architecture, not tuned mid-project:

| Model | Default Optimizer |
|-------|-----------------|
| CNN (vision, from scratch) | SGD + Momentum (0.9) + Nesterov |
| Transformer (NLP, vision) | AdamW |
| Fine-tuning any pretrained | AdamW |
| Reinforcement learning | Adam or RMSProp |
| Very large models (memory issue) | Adafactor or Lion |

The common mistake is using Adam for CNNs (where SGD+Momentum generalizes better) or SGD for transformers (where it's hard to tune and underperforms AdamW).

---

## Architecture Depth

Least frequently tuned in practice — most projects use a standard pretrained architecture.

If you are making this choice: deeper = more capacity but harder to train. For transfer learning, use the largest model your compute budget allows and regularize aggressively. For training from scratch on small datasets, use the smallest model that achieves acceptable accuracy — more capacity means more overfitting risk.

---

## Practical Tuning Strategy

**Step 1 — Set reasonable defaults:**
```
optimizer  = AdamW (or SGD+Momentum for CNNs)
lr         = 1e-3 (Adam) or 0.1 (SGD)   ← to be tuned
weight_decay = 1e-4
scheduler  = CosineAnnealingLR or Cosine+Warmup
dropout    = architecture default
grad_clip  = 1.0 (transformers) or off (CNNs)
```

**Step 2 — Find the LR:** Run LR range test. Set LR to ~1/5 of the point of steepest descent.

**Step 3 — Tune weight decay:** Run 2–3 short experiments with `[1e-5, 1e-4, 1e-2]`. Pick best val performance.

**Step 4 — Adjust regularization:** If overfitting → increase dropout/weight decay. If underfitting → decrease both.

**Step 5 — Full run:** Final training with tuned hyperparameters, correct epoch count, and early stopping or fixed schedule.

---

## Quick Reference: Common Recipes

**ResNet-50 on ImageNet:**
```
lr=0.1, momentum=0.9, weight_decay=1e-4, batch=256, epochs=90
scheduler=MultiStepLR(milestones=[30,60,90], gamma=0.1)
```

**BERT Fine-tuning:**
```
lr=2e-5, weight_decay=0.01, warmup=6% of steps
scheduler=Cosine+Warmup, batch=32, epochs=3–5, grad_clip=1.0
```

**ViT from Scratch (ImageNet):**
```
lr=1e-3, weight_decay=0.1, warmup=10k steps, batch=1024, epochs=300
scheduler=Cosine+Warmup, grad_clip=1.0, dropout=0.1, drop_path=0.1
```

**GPT-style Pretraining:**
```
lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95), warmup=2000 steps
scheduler=Cosine+Warmup, grad_clip=1.0, batch=512
```