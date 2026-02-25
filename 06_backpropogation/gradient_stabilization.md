# Gradient Stabilization Techniques

> **Complete guide to preventing gradient explosion and enabling large-batch training**

---

## Table of Contents

1. [Why Gradient Stabilization?](#why-gradient-stabilization)
2. [Gradient Clipping](#gradient-clipping)
3. [Gradient Accumulation](#gradient-accumulation)
4. [Other Stabilization Techniques](#other-stabilization-techniques)
5. [Practical Guidelines](#practical-guidelines)
6. [Common Problems & Solutions](#common-problems--solutions)

---

## Why Gradient Stabilization?

### The Problem: Unstable Training

**Symptoms:**
- Loss becomes NaN
- Weights explode to infinity
- Training diverges
- Model outputs nonsense

**Root Causes:**
1. **Exploding Gradients**: Gradients become very large
2. **Vanishing Gradients**: Gradients become very small
3. **Limited GPU Memory**: Can't use large batch sizes

```
Unstable Training:
═══════════════════════════════════════════════════════════
Loss: 2.5 → 2.3 → 2.0 → 1.8 → 357.2 → NaN → Training crashed!
                                ↑
                        Gradient exploded
```

### Why Gradients Explode

**Multiplicative Effect:**
```
Deep network with L layers:
∂L/∂W₁ = ∂L/∂aₗ × ∂aₗ/∂aₗ₋₁ × ... × ∂a₂/∂a₁ × ∂a₁/∂W₁
         └──────────────────┬─────────────────┘
         Product of L terms

If each gradient > 1:
  Example: 1.1^50 = 117  (exploded after 50 layers!)
  
If each gradient < 1:
  Example: 0.9^50 = 0.005 (vanished after 50 layers!)
```

**Common Causes:**
- RNNs with long sequences
- Very deep networks
- Large learning rates
- Poor initialization
- Certain activation functions (e.g., sigmoid)

---

## Gradient Clipping

### The Core Idea

**Idea:** If gradient becomes too large, scale it down to a maximum threshold.

```
Original gradient: [10, 25, 30, 100]  (some very large values)
After clipping:    [1.5, 3.8, 4.5, 15] (scaled down proportionally)
```

**Analogy:**
- Like a speed limiter on a car
- Allows safe "driving" without crashing
- Doesn't stop learning, just prevents sudden jumps

### Types of Gradient Clipping

#### **1. Gradient Norm Clipping (Most Common)**

**Idea:** Scale gradient if its norm exceeds threshold.

```
||g|| = sqrt(g₁² + g₂² + ... + gₙ²)  (L2 norm)

If ||g|| > max_norm:
    g = g × (max_norm / ||g||)
```

**Visual:**
```
Before Clipping:           After Clipping:
                          
    ↑                          ↑
    │    g (large)             │    g' (clipped)
    │   /                      │   /
    │  /                       │  /
    │ /                        │ /
    │/                         │/
    └─────→                    └─────→
    
||g|| = 10                 ||g'|| = 5 (max_norm)
Direction preserved, magnitude reduced
```

**Code:**
```python
import torch
import torch.nn as nn

# Most common: clip by global norm
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=1.0  # Typical value: 0.5 - 5.0
)

# Complete training loop
for data, target in dataloader:
    optimizer.zero_grad()
    
    output = model(data)
    loss = criterion(output, target)
    
    loss.backward()
    
    # Clip gradients BEFORE optimizer.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

**How it works:**
```python
# Under the hood
def clip_grad_norm_(parameters, max_norm):
    parameters = list(parameters)
    
    # Compute total norm
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Scale gradients if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
```

#### **2. Gradient Value Clipping**

**Idea:** Clip each gradient value individually to a range.

```
grad[i] = clip(grad[i], min_value, max_value)
```

**Code:**
```python
# Clip each gradient element to [-5, 5]
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)
```

**When to use:**
- Less common than norm clipping
- Useful when you want hard bounds on individual gradients
- Can distort gradient direction (unlike norm clipping)

```
Norm Clipping:              Value Clipping:
Preserves direction         May change direction
  
     ↑                           ↑
     │  g                        │
     │ /                         │  g'
     │/                          │ ╱
     └─────→                     └─────→
     
Scale entire vector         Clip each element independently
```

#### **3. Adaptive Gradient Clipping (AGC)**

**Idea:** Clip gradient relative to the parameter norm (not absolute threshold).

```
For each layer:
    clip_norm = λ × ||W|| / ||∇W||
    ∇W = min(1, clip_norm) × ∇W
```

**Benefits:**
- Automatically adapts to layer scale
- Works better for very large models
- Used in modern architectures (NFNets)

**Code:**
```python
def adaptive_clip_grad(parameters, clip_factor=0.01):
    for p in parameters:
        if p.grad is None:
            continue
        
        p_norm = p.norm()
        g_norm = p.grad.norm()
        
        max_norm = p_norm * clip_factor
        clip_coef = max_norm / (g_norm + 1e-6)
        
        if clip_coef < 1:
            p.grad.mul_(clip_coef)
```

### When to Use Gradient Clipping

#### **Always Use For:**

✅ **RNNs, LSTMs, GRUs**
```python
# RNN training - ALWAYS clip!
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
**Why:** RNNs are notorious for exploding gradients

✅ **GANs**
```python
# GAN training - stabilizes both generator and discriminator
loss_G.backward()
torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
optimizer_G.step()
```
**Why:** Adversarial training is inherently unstable

✅ **Very Deep Networks (50+ layers)**
```python
# Deep ResNets, Transformers
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
**Why:** Long gradient paths can amplify errors

✅ **When You See NaN Losses**
```python
# Emergency fix for unstable training
if torch.isnan(loss):
    print("NaN detected! Using gradient clipping.")
    # Add clipping or reduce learning rate
```

#### **May Not Need For:**

⚠️ **Shallow Networks (< 10 layers)**
⚠️ **Well-behaved CNNs with BatchNorm**
⚠️ **When using Adam/RMSprop** (they have built-in adaptive scaling)

### Choosing max_norm

**Guidelines:**

```
Very Conservative:  max_norm = 0.5
  - Use when training is very unstable
  - Slower learning but very safe
  
Standard (Default): max_norm = 1.0
  - Good starting point for most models
  - Works for RNNs, Transformers
  - Most common choice
  
Permissive:        max_norm = 5.0
  - Use for stable models
  - Allows faster learning
  - GANs often use higher values
  
Very Permissive:   max_norm = 10.0+
  - Rarely needed
  - Only if gradients naturally large
```

**How to tune:**
```python
# Monitor gradient norms during training
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"Gradient norm: {total_norm:.4f}")

# If often > 10: Reduce max_norm or learning rate
# If always < 0.1: Increase max_norm or learning rate
```

### Effects of Gradient Clipping

**Visualization:**
```
Without Clipping:
═══════════════════════════════════════════════════════════
Loss: 2.5 → 2.3 → 2.0 → 357.2 → NaN
            ↑            ↑
      Normal update  Huge gradient caused explosion

With Clipping (max_norm=1.0):
═══════════════════════════════════════════════════════════
Loss: 2.5 → 2.3 → 2.0 → 1.9 → 1.8 → 1.7 → ...
            ↑            ↑
      Normal update  Large gradient clipped, stable update
```

**Trade-offs:**
```
✅ Pros:
  - Prevents training collapse
  - Enables training of RNNs and GANs
  - Simple to implement
  - Small computational overhead

❌ Cons:
  - May slow convergence (if too aggressive)
  - Introduces hyperparameter (max_norm)
  - Doesn't fix root cause (just symptom)
```

---

## Gradient Accumulation

### The Problem: Limited GPU Memory

**Scenario:**
```
Want:      batch_size = 256 (good for training)
Reality:   GPU can only fit batch_size = 32
Result:    Poor training, noisy gradients
```

**Why large batches matter:**
- More stable gradient estimates
- Better generalization (up to a point)
- Enables higher learning rates
- Reduces training variance

### The Solution: Accumulate Gradients

**Idea:** Split large batch into small mini-batches, accumulate gradients, then update.

```
Large Batch (256):
═══════════════════════════════════════════════════════════
[────────── 256 samples ──────────]
         ↓
   Compute gradient
         ↓
     Update weights
═══════════════════════════════════════════════════════════

Gradient Accumulation (32 × 8):
═══════════════════════════════════════════════════════════
[32] → grad₁
[32] → grad₂  } Accumulate
[32] → grad₃  }
[32] → grad₄  } 
[32] → grad₅  } 8 steps
[32] → grad₆  }
[32] → grad₇  }
[32] → grad₈  }
    ↓
grad_total = (grad₁ + grad₂ + ... + grad₈) / 8
    ↓
Update weights (once)
═══════════════════════════════════════════════════════════
Same result as batch_size=256!
```

### Implementation

#### **Method 1: Manual Accumulation**

```python
accumulation_steps = 8  # Effective batch = 32 × 8 = 256
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Scale loss (IMPORTANT!)
    loss = loss / accumulation_steps
    
    # Backward pass (accumulates gradients)
    loss.backward()
    
    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Key Points:**
1. Divide loss by `accumulation_steps` (average gradients)
2. Don't zero gradients until after update
3. Call `optimizer.step()` only after accumulating

#### **Method 2: With Gradient Clipping**

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        # Clip gradients BEFORE optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
```

#### **Method 3: With Mixed Precision**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    # Forward in FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
    
    # Backward with scaling
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        # Clip gradients
        scaler.unscale_(optimizer)  # Unscale before clipping!
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Visual Comparison

```
Standard Training (batch=256):
═══════════════════════════════════════════════════════════
Iteration 1:  [────── 256 samples ──────] → Update
Iteration 2:  [────── 256 samples ──────] → Update
Iteration 3:  [────── 256 samples ──────] → Update

GPU Memory: High (might not fit)
Updates per epoch: N
═══════════════════════════════════════════════════════════

Gradient Accumulation (32×8):
═══════════════════════════════════════════════════════════
Iteration 1:  [32] → [32] → [32] → [32] → [32] → [32] → [32] → [32] → Update
Iteration 2:  [32] → [32] → [32] → [32] → [32] → [32] → [32] → [32] → Update
Iteration 3:  [32] → [32] → [32] → [32] → [32] → [32] → [32] → [32] → Update

GPU Memory: Low (fits easily)
Updates per epoch: N (same number of updates!)
Effective batch size: 256 (same as standard!)
═══════════════════════════════════════════════════════════
```

### When to Use Gradient Accumulation

✅ **GPU Memory Limited**
```
Want batch_size=128, GPU can only fit 32
→ Use accumulation_steps=4
```

✅ **Training Large Models**
```
BERT, GPT, ViT training
→ Often requires accumulation
```

✅ **Limited Hardware**
```
Training on single GPU instead of multiple
→ Simulate distributed training
```

✅ **Stable Large-Batch Training**
```
Some tasks need large batches for stability
→ Contrastive learning (SimCLR, CLIP)
```

### Choosing accumulation_steps

```
Current batch_size: 32
Desired batch_size: 256
accumulation_steps = 256 / 32 = 8
```

**Guidelines:**
```
Small (2-4):
  - Slight memory savings
  - Minimal slowdown
  - Use when close to limit
  
Medium (8-16):
  - Standard for large models
  - Good memory savings
  - Acceptable slowdown
  
Large (32-64):
  - Maximum memory savings
  - Significant slowdown
  - Use only when necessary
```

### Effects on Training

**Advantages:**
```
✅ Can train larger models
✅ Larger effective batch size
✅ More stable gradients
✅ Better performance (up to a point)
✅ Same end result as large batch
```

**Disadvantages:**
```
❌ Slower training (more forward passes per update)
❌ Batch norm statistics less accurate (small batches)
❌ Longer to see progress (fewer updates per epoch)
```

**Performance Impact:**
```
Batch size: 32 → 256 (8× accumulation)
Training time: +20-40% slower
Memory usage: Same as batch_size=32
Final accuracy: Similar to batch_size=256
```

---

## Other Stabilization Techniques

### 1. Gradient Centralization

**Idea:** Center gradients to zero mean.

```python
def centralized_gradient(grad):
    # Center each gradient
    grad = grad - grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)
    return grad

# Apply during backward pass
for p in model.parameters():
    if p.grad is not None:
        p.grad = centralized_gradient(p.grad)
```

**Benefits:**
- More stable training
- Faster convergence
- Works well with SGD

### 2. Warmup

**Idea:** Start with small learning rate, gradually increase.

```python
# Linear warmup
def get_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr

# Usage with scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda step: min(1.0, step / warmup_steps)
)
```

**Why it helps:**
- Initial gradients can be very large
- Warmup prevents early instability
- Common in Transformer training

### 3. Weight Decay

**Idea:** Regularize weights, preventing them from growing too large.

```python
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.001, 
    weight_decay=0.01  # L2 regularization
)
```

### 4. Batch Normalization

**Idea:** Normalize activations, which indirectly stabilizes gradients.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)  # ← Stabilizes gradients
```

---

## Practical Guidelines

### Complete Training Setup (Best Practices)

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Model
model = MyLargeModel().cuda()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=0.01
)

# Learning rate scheduler with warmup
warmup_steps = 1000
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda step: min(1.0, step / warmup_steps)
)

# Mixed precision
scaler = GradScaler()

# Gradient accumulation
accumulation_steps = 8

# Training loop
model.train()
optimizer.zero_grad()

for epoch in range(num_epochs):
    for i, (data, target) in enumerate(dataloader):
        # Forward pass with mixed precision
        with autocast():
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=1.0
            )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Learning rate scheduler
            scheduler.step()
```

### Debugging Unstable Training

```python
def check_gradients(model):
    """Debug gradient issues"""
    total_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # Check for NaN
            if torch.isnan(p.grad).any():
                print(f"NaN gradient in {name}")
            
            # Check for very large gradients
            if param_norm > 100:
                print(f"Large gradient in {name}: {param_norm:.2f}")
    
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm:.4f}")
    return total_norm

# Use during training
for i, (data, target) in enumerate(dataloader):
    loss.backward()
    
    grad_norm = check_gradients(model)
    
    if grad_norm > 10:
        print(f"Warning: Large gradient at step {i}")
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## Common Problems & Solutions

### Problem 1: Loss Becomes NaN

**Symptoms:**
```
Epoch 1: Loss = 2.5
Epoch 2: Loss = 2.3
Epoch 3: Loss = nan
```

**Causes:**
- Exploding gradients
- Division by zero
- Log of negative number
- Learning rate too high

**Solutions:**
```python
# 1. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Lower

# 3. Use mixed precision with loss scaling
scaler = GradScaler()

# 4. Check for NaN in data
assert not torch.isnan(loss), "NaN in loss!"
```

### Problem 2: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# 1. Reduce batch size
batch_size = 16  # Down from 32

# 2. Use gradient accumulation
accumulation_steps = 4  # Effective batch = 16 × 4 = 64

# 3. Use gradient checkpointing (see efficiency_techniques.md)
model.gradient_checkpointing_enable()

# 4. Use mixed precision
with autocast():
    output = model(data)
```

### Problem 3: Training Unstable

**Symptoms:**
```
Loss: 2.5 → 2.0 → 2.3 → 1.8 → 2.5 → 2.1 → ...
      (jumping around, not converging)
```

**Solutions:**
```python
# 1. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# 2. Reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 3. Add warmup
for step in range(warmup_steps):
    lr = base_lr * (step / warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 4. Use larger batch size (via accumulation)
accumulation_steps = 8
```

### Problem 4: Slow Convergence with Clipping

**Symptoms:**
```
With clipping: Loss decreases very slowly
Without clipping: Loss decreases fast but unstable
```

**Solutions:**
```python
# 1. Increase max_norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 2. Adjust learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Higher

# 3. Monitor how often clipping triggers
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if grad_norm > 1.0:
    print(f"Clipped: {grad_norm:.2f} → 1.0")
# If clipping too often, increase max_norm
```

---

## Key Takeaways

### Essential Patterns

**1. RNN/LSTM Training:**
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ALWAYS
optimizer.step()
```

**2. Large Model Training:**
```python
accumulation_steps = 8
for i, (data, target) in enumerate(dataloader):
    loss = criterion(model(data), target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**3. Complete Stable Training:**
```python
scaler = GradScaler()
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    with autocast():
        loss = criterion(model(data), target) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Quick Reference

| Technique | Purpose | When to Use | Typical Values |
|-----------|---------|-------------|----------------|
| **Gradient Clipping** | Prevent explosion | RNNs, GANs, deep nets | max_norm=1.0 |
| **Gradient Accumulation** | Simulate large batch | Memory limited | steps=4-16 |
| **Warmup** | Stable start | Transformers | 1000-10000 steps |
| **Weight Decay** | Regularization | Most models | 0.01-0.1 |

---

*See also: [Efficiency Techniques](./efficiency_techniques.md) for memory optimization*