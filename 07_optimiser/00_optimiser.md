# Optimizers in Deep Learning

> **A quick guide to understanding what optimizers are and how they're organized**

---

## What is an Optimizer?

An **optimizer** is the algorithm that **updates the model's weights** to minimize the loss function.

### The Core Process

```
Training Loop:
1. Forward pass → Get predictions
2. Calculate loss → How wrong are we?
3. Backward pass → Compute gradients (∂Loss/∂Weight)
4. Optimizer → Update weights using gradients ← THIS IS THE OPTIMIZER
5. Repeat
```

**Simple analogy:**
- **Gradient** = Direction to go (uphill/downhill)
- **Optimizer** = How you walk (step size, momentum, adaptive steps)

### The Basic Update Rule

```
Weight_new = Weight_old - learning_rate × gradient

Where:
  - Weight_old: Current parameter value
  - gradient: How to change it (from backprop)
  - learning_rate: How big a step to take
  - Weight_new: Updated parameter value
```

---

## Why Do We Need Different Optimizers?

**The problem:** Gradient descent is simple but has issues:

```
Issues with basic gradient descent:
❌ Gets stuck in local minima
❌ Slow convergence
❌ Same learning rate for all parameters
❌ Sensitive to learning rate choice
❌ Oscillates in valleys

Different optimizers solve different problems!
```

---

## Optimizer Taxonomy

```
├── Optimizer
│     │
│     ├── SGD family               [Momentum-based]
│     │   ├── SGD                  Basic gradient descent
│     │   ├── Momentum             Add velocity (physics-based)
│     │   └── Nesterov             Look-ahead momentum
│     │
│     ├── Adaptive                 [Automatic learning rate adjustment]
│     │   ├── Adam                 Most popular default
│     │   ├── AdamW                Adam + proper weight decay (BERT/GPT)
│     │   ├── RMSProp              Good for RNNs
│     │   └── AdaGrad              Rarely used (LR decays too fast)
│     │
│     ├── Modern                   [Latest research, 2020+]
│     │   ├── Lion                 Simpler than Adam
│     │   ├── Adafactor            Memory-efficient (T5)
│     │   └── LAMB                 Large-batch training
│     │
│     ├── Weight Decay             [Regularization]
│     │   └── L2 penalty on weights (prevents overfitting)
│     │
│     └── Learning Rate Schedulers ⭐⭐⭐ [Dynamic LR adjustment]
│           ├── StepLR             Drop LR every N epochs
│           ├── CosineAnnealingLR  🔥 Smooth decay (most popular)
│           ├── OneCycleLR         🔥 Super-convergence
│           ├── ReduceLROnPlateau  Drop when stuck
│           └── Warmup variants    ⭐ Essential for Transformers
```

---

## Visual: Optimizer Families

### **1. SGD Family (Momentum-Based)**

```
Idea: Build up velocity like a ball rolling downhill

    SGD (Basic)
       ↓
    Add momentum → SGD + Momentum
       ↓
    Look ahead → Nesterov Momentum

Characteristics:
  ✓ Simple and fast
  ✓ Works well with large learning rates
  ✗ Requires careful LR tuning
  ✗ Same LR for all parameters
```

**Visual representation:**
```
SGD path:
    ╱╲  ╱╲  ╱╲      (zig-zag, slow)
   ╱  ╲╱  ╲╱  ╲

Momentum path:
    ╱───────────╲   (smooth, faster)
   ╱             ╲
```

---

### **2. Adaptive Family (Per-Parameter Learning Rates)**

```
Idea: Different learning rates for each parameter

    AdaGrad (historical, too aggressive)
       ↓
    RMSProp (fixes AdaGrad decay)
       ↓
    Adam (RMSProp + Momentum) ← Most popular
       ↓
    AdamW (Adam + proper weight decay) ← Current best practice

Characteristics:
  ✓ Adapts to each parameter
  ✓ Less sensitive to initial LR
  ✓ Works out-of-the-box
  ✗ More memory (stores moving averages)
  ✗ Can overfit without weight decay
```

**Visual representation:**
```
Parameter space with different curvature:

Flat dimension: ─────────── (needs large LR)
Steep dimension: │││││││││││ (needs small LR)

Adaptive optimizers adjust automatically!
```

---

### **3. Modern Family (Research & Specialized)**

```
Lion (2023):      Simpler than Adam, sign-based updates
Adafactor (2018): Memory-efficient for huge models (T5)
LAMB (2019):      Large-batch training (BERT-scale)

When to use:
  → Cutting-edge research
  → Specific constraints (memory, batch size)
  → Most people stick to AdamW
```

---

### **4. Weight Decay (Regularization)**

```
Idea: Penalize large weights

Weight_new = Weight_old - lr × gradient - lr × weight_decay × Weight_old
                                          └──────────┬──────────┘
                                          Pulls weights toward zero

Why:
  ✓ Prevents overfitting
  ✓ Simpler models (smaller weights)
  
Typical values: 0.01, 0.001, 0.0001
```

**Visual:**
```
Without weight decay:    With weight decay:
  Weights can be large      Weights stay small
  ⬤⬤⬤⬤⬤⬤⬤⬤                  ⚫⚫⚫⚫
  Might overfit             Better generalization
```

---

### **5. Learning Rate Schedulers (Dynamic Adjustment)**

```
Problem: Fixed learning rate is suboptimal
  → Start high (fast learning)
  → End low (fine-tuning)

Solution: Adjust LR during training
```

**Visual: LR over time**

```
StepLR:
LR │─────┐
   │     └────┐
   │          └────┐
   └───────────────────> Epochs
   (Drops every N epochs)

CosineAnnealingLR:
LR │╲
   │ ╲___
   │     ╲___
   │         ╲___
   └───────────────────> Epochs
   (Smooth cosine decay)

OneCycleLR:
LR │    ╱╲
   │   ╱  ╲
   │  ╱    ╲___
   │ ╱         ╲___
   └───────────────────> Epochs
   (Up then down)

Warmup + Cosine:
LR │  ╱─╲
   │ ╱   ╲___
   │╱        ╲___
   └───────────────────> Epochs
   (Gradual start + decay)
```

---

## When to Use What?

### **Quick Decision Tree**

```
Starting a new project?
└─> Use AdamW (lr=3e-4, weight_decay=0.01)
    + CosineAnnealingLR
    Works 90% of the time

Training Transformers (BERT, GPT)?
└─> AdamW + Warmup + Cosine/Linear decay
    Industry standard

Training CNNs (ResNet, EfficientNet)?
└─> AdamW + CosineAnnealingLR
    or SGD + Momentum (if you want to tune more)

Training RNNs/LSTMs?
└─> Adam or RMSProp
    Handles noisy gradients

Very large batches (>4k)?
└─> LAMB
    Designed for this

Need extreme speed?
└─> OneCycleLR with AdamW
    Fastest convergence
```

---

## The Complete Picture

### **Optimizer Evolution Timeline**

```
1950s: SGD (Basic gradient descent)
         ↓
1980s: Momentum (Add velocity)
         ↓
1990s: RMSProp (Adaptive per-parameter)
         ↓
2014:  Adam (RMSProp + Momentum) ← Revolution!
         ↓
2017:  AdamW (Proper weight decay) ← Current standard
         ↓
2019:  LAMB (Large batch)
         ↓
2023:  Lion (Simplification)
```

### **Modern Best Practice (2024)**

```python
# This is what most people use today:

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # Default learning rate
    weight_decay=0.01  # Regularization
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)

# For Transformers, add warmup:
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)
```

---

## Comparison Table

| Family | Example | Memory | Speed | Tuning | When to Use |
|--------|---------|--------|-------|--------|-------------|
| **SGD** | SGD + Momentum | Low | Fast | Hard | CNNs if you want to tune |
| **Adaptive** | AdamW | High | Fast | Easy | Default choice (90% of cases) |
| **Modern** | Lion | Medium | Fast | Easy | Research, cutting-edge |
| **Schedulers** | CosineAnnealingLR | None | None | Easy | Always! Pair with optimizer |

---

## Key Concepts

### **1. Learning Rate (Most Important!)**

```
Too high:  Training diverges, loss → NaN
Too low:   Training too slow, never converges
Just right: Fast and stable convergence

Typical ranges:
  AdamW:     1e-4 to 1e-3
  SGD:       1e-2 to 1e-1 (higher than adaptive)
  Transformers: 1e-5 to 5e-5
```

### **2. Adaptive vs Non-Adaptive**

```
Non-Adaptive (SGD, Momentum):
  ✓ Simple
  ✓ Less memory
  ✗ Need to tune LR carefully
  
Adaptive (Adam, AdamW):
  ✓ Auto-adjusts per parameter
  ✓ Works out-of-box
  ✗ More memory
```

### **3. Why Schedulers Matter**

```
Without scheduler:
  Loss: 2.5 → 1.8 → 1.2 → 0.9 → 0.7 → 0.65 → stuck at 0.65

With scheduler:
  Loss: 2.5 → 1.8 → 1.2 → 0.9 → 0.7 → 0.5 → 0.3 ← Better!
                                  ↑
                          LR decreases, fine-tunes
```

---

## Summary

### **What is an Optimizer?**
The algorithm that updates weights using gradients to minimize loss.

### **Why Different Optimizers?**
Different problems need different solutions (momentum, adaptive rates, large batches).

### **What to Use?**
- **Default**: AdamW + CosineAnnealingLR
- **Transformers**: AdamW + Warmup + Cosine
- **CNNs**: AdamW or SGD + Momentum

### **Most Important Hyperparameter?**
**Learning Rate** - Get this right first, everything else is secondary.

### **Must-Have Addition?**
**Learning Rate Scheduler** - Without it, you're leaving 20-30% performance on the table!

---

## The Essential Pattern

```python
# 95% of deep learning uses this pattern:

# 1. Choose optimizer (AdamW is default)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)

# 2. Choose scheduler (CosineAnnealing is popular)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

# 3. Training loop
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # Update learning rate
```

**That's it! Master this pattern and you're 90% there.**

---

*Remember: Start simple (AdamW + scheduler), only optimize if needed.*