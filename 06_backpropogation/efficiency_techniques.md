# Efficiency Techniques for Backpropagation

> **Complete guide to memory optimization and speed improvements in training**

---

## Table of Contents

1. [Why Efficiency Matters](#why-efficiency-matters)
2. [Gradient Checkpointing](#gradient-checkpointing)
3. [Mixed Precision Training (AMP)](#mixed-precision-training)
4. [Combining Techniques](#combining-techniques)
5. [Practical Guidelines](#practical-guidelines)
6. [Performance Benchmarks](#performance-benchmarks)

---

## Why Efficiency Matters

### The Resource Challenge

**Modern deep learning models:**
```
GPT-3:      175 billion parameters
BERT-Large: 340 million parameters
ViT-Large:  300 million parameters

Problem: 
  - High memory usage
  - Slow training
  - Expensive hardware required
```

### Memory Bottleneck in Training

**What consumes memory during training?**

```
Total Memory = Model Weights + Activations + Gradients + Optimizer States

Example (BERT-Base, batch=32):
═══════════════════════════════════════════════════════════
Model weights:        110M params × 4 bytes = 440 MB
Optimizer states:     110M × 8 bytes        = 880 MB  (Adam: momentum + variance)
Gradients:            110M × 4 bytes        = 440 MB
Activations:          ~12 layers × batch    = 3-8 GB  ← LARGEST!
═══════════════════════════════════════════════════════════
Total: ~5-10 GB for single batch!
```

**Activations** are the memory bottleneck!

### Forward vs Backward Memory

```
Forward Pass:
═══════════════════════════════════════════════════════════
Input → Layer1 → Layer2 → Layer3 → ... → Output
        ↓save    ↓save    ↓save
        a₁       a₂       a₃
        
Must SAVE all intermediate activations for backward pass!
Memory: O(L) where L = number of layers
═══════════════════════════════════════════════════════════

Backward Pass:
═══════════════════════════════════════════════════════════
Uses saved activations to compute gradients
∂L/∂W₃ needs a₂
∂L/∂W₂ needs a₁
etc.
═══════════════════════════════════════════════════════════
```

**The Trade-off:**
- **Speed:** Save all activations (fast backward, high memory)
- **Memory:** Recompute activations (slow backward, low memory)

---

## Gradient Checkpointing

### The Core Idea

**Standard Training:**
```
Forward:  Save ALL activations
Backward: Use saved activations

Memory: High ✅ Fast
```

**Gradient Checkpointing:**
```
Forward:  Save ONLY checkpoint activations
Backward: Recompute other activations when needed

Memory: Low ✅ Slower
```

### Visual Explanation

```
Standard (Save Everything):
═══════════════════════════════════════════════════════════
Forward:
x → f₁(x) → f₂ → f₃ → f₄ → f₅ → f₆ → output
    ↓save   ↓    ↓    ↓    ↓    ↓
    a₁      a₂   a₃   a₄   a₅   a₆

Backward:
    Use a₁  Use a₂ ...

Memory: 6 activations stored
═══════════════════════════════════════════════════════════

Gradient Checkpointing (Save Every 2nd):
═══════════════════════════════════════════════════════════
Forward:
x → f₁(x) → f₂ → f₃ → f₄ → f₅ → f₆ → output
    ↓save   X    ↓    X    ↓    X
    a₁      ✗    a₃   ✗    a₅   ✗

Backward for f₆:
  Need a₅ ✅ (saved)
  
Backward for f₅:
  Need a₄ ✗ (not saved)
  → Recompute: x → f₁ → f₂ → f₃ → f₄ → a₄ ✅
  
Backward for f₄:
  Need a₃ ✅ (saved)

Memory: 3 activations stored (50% reduction!)
Time: +20-30% (recomputation overhead)
═══════════════════════════════════════════════════════════
```

### Mathematical Foundation

**Standard backprop:**
```
Memory:  O(n) where n = number of layers
Time:    O(n) forward + O(n) backward = O(n)
```

**With checkpointing (k checkpoints):**
```
Memory:  O(√n) or O(n/k) depending on strategy
Time:    O(n) forward + O(n) backward + O(n/k) recomputation = O(n)
```

**Sweet spot:** Save every √n layers
- Memory: O(√n) 
- Time: O(√n) recomputation overhead

### Implementation

#### **Method 1: PyTorch checkpoint**

```python
import torch
from torch.utils.checkpoint import checkpoint

# Wrap your forward computation
def my_layer(x):
    return layer1(layer2(layer3(x)))

# Without checkpointing (standard)
output = my_layer(input)

# With checkpointing (memory efficient)
output = checkpoint(my_layer, input)
```

**How it works:**
```python
# Under the hood
def checkpoint(function, *args):
    # Forward pass: DON'T save intermediate activations
    with torch.no_grad():
        output = function(*args)
    
    # Save inputs and function for backward
    # When backward is called:
    #   1. Recompute forward (with grad tracking this time)
    #   2. Then compute gradients
    
    return output
```

#### **Method 2: Automatic for Transformers**

```python
from transformers import BertModel

# Load model
model = BertModel.from_pretrained('bert-base-uncased')

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Training as usual
output = model(input_ids)
loss = criterion(output, target)
loss.backward()  # ← Activations recomputed automatically
```

#### **Method 3: Manual Checkpointing**

```python
class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 512)
        
        self.checkpoint_layers = [self.layer1, self.layer3]  # Checkpoint every 2
    
    def forward(self, x):
        # Layer 1 (checkpointed)
        x = checkpoint(lambda z: F.relu(self.layer1(z)), x)
        
        # Layer 2 (not checkpointed)
        x = F.relu(self.layer2(x))
        
        # Layer 3 (checkpointed)
        x = checkpoint(lambda z: F.relu(self.layer3(z)), x)
        
        # Layer 4 (not checkpointed)
        x = F.relu(self.layer4(x))
        
        return x
```

### Checkpointing Strategies

#### **Strategy 1: Uniform Checkpointing**

Save every k-th layer:
```
12 layers, checkpoint every 3:
Save: Layer 0, 3, 6, 9
Skip: Layers 1,2,4,5,7,8,10,11

Memory: O(n/k)
Best for: Uniform layer costs
```

#### **Strategy 2: Square Root Checkpointing**

Save √n layers:
```
16 layers, √16 = 4:
Save: Layers 0, 4, 8, 12

Memory: O(√n)
Best for: Optimal time-memory trade-off
```

#### **Strategy 3: Selective Checkpointing**

Save expensive layers only:
```
Checkpoint:
  - Attention layers (expensive)
  - Large matrix multiplications
  
Don't checkpoint:
  - Activations (cheap)
  - Normalization (cheap)
```

### When to Use Gradient Checkpointing

✅ **Use When:**

**1. Training Very Large Models**
```python
# BERT-Large, GPT-2, ViT-Large
model = BertModel.from_pretrained('bert-large-uncased')
model.gradient_checkpointing_enable()  # ← Essential!
```

**2. GPU Memory Maxed Out**
```
RuntimeError: CUDA out of memory
↓
Enable gradient checkpointing
↓
Training succeeds!
```

**3. Want Larger Batch Sizes**
```
Current: batch_size=8  (barely fits)
Goal:    batch_size=32 (doesn't fit)

Solution:
  Enable checkpointing → Can fit batch_size=32
```

**4. Long Sequences**
```
Sequence length: 512 tokens
Memory per sequence: Very high
↓
Enable checkpointing → Can handle longer sequences
```

❌ **Don't Use When:**

**1. Small Models**
```
Model < 50M parameters
→ Overhead not worth it
```

**2. Memory Not an Issue**
```
Plenty of GPU RAM available
→ Standard training is faster
```

**3. Need Maximum Speed**
```
Training time is critical
Memory is not a constraint
→ Avoid checkpointing overhead
```

### Memory Savings

**Example: BERT-Base (12 layers)**

```
Without Checkpointing:
═══════════════════════════════════════════════════════════
Forward: Save 12 activations
Memory: ~8 GB for batch=32
═══════════════════════════════════════════════════════════

With Checkpointing (every 3 layers):
═══════════════════════════════════════════════════════════
Forward: Save 4 activations
Memory: ~3 GB for batch=32 (62% reduction!)
═══════════════════════════════════════════════════════════

Trade-off: +20-30% training time
```

**Real-world examples:**

| Model | Without | With Checkpointing | Savings |
|-------|---------|-------------------|---------|
| BERT-Base | 8 GB | 3 GB | 62% |
| BERT-Large | 24 GB | 8 GB | 67% |
| GPT-2 | 12 GB | 4 GB | 67% |
| ViT-Large | 18 GB | 6 GB | 67% |

---

## Mixed Precision Training

### What is Mixed Precision?

**Floating Point Types:**
```
FP32 (Float32):
  - 32 bits per number
  - High precision
  - Standard in deep learning
  - Range: ±3.4 × 10³⁸

FP16 (Float16):
  - 16 bits per number  (half size!)
  - Lower precision
  - 2× faster on modern GPUs
  - Range: ±6.5 × 10⁴  (much smaller!)

BF16 (BFloat16):
  - 16 bits per number
  - Same range as FP32
  - Less precision than FP32
  - Better than FP16 for deep learning
```

**Mixed Precision Strategy:**
```
Forward pass:     FP16  (fast, less memory)
Backward pass:    FP16  (fast, less memory)
Optimizer:        FP32  (stable, accurate)
Master weights:   FP32  (prevent underflow)
```

### Why Mixed Precision Works

**GPU Performance:**
```
NVIDIA A100:
  FP32: 19.5 TFLOPS
  FP16: 312 TFLOPS  ← 16× faster!

Memory Bandwidth:
  FP32: Need to move 4 bytes per number
  FP16: Need to move 2 bytes per number  ← 2× less data!
```

**Memory Savings:**
```
Model weights:     2× less (FP16 vs FP32)
Activations:       2× less
Gradients:         2× less
Total saving:      ~2× memory reduction!
```

### The Challenge: Numerical Stability

**Problem 1: Underflow**
```
FP16 range: 6.5 × 10⁻⁵ to 6.5 × 10⁴

Small gradients (< 6.5 × 10⁻⁵) → Become 0!

Example:
  Gradient = 0.00001 (FP32)
  → 0.0 (FP16, underflow)
  → No learning!
```

**Problem 2: Overflow**
```
Large values (> 65504) → Become Inf!

Example:
  Activation = 100000 (FP32)
  → Inf (FP16, overflow)
  → NaN loss!
```

**Solution: Loss Scaling**
```
1. Scale loss UP before backward:
   loss_scaled = loss × scale_factor  (e.g., × 1024)
   
2. Gradients are scaled UP:
   grad_scaled = grad × scale_factor
   
3. Scale gradients DOWN before optimizer:
   grad = grad_scaled / scale_factor
   
Result: Gradients stay in FP16 range, no underflow!
```

### Automatic Mixed Precision (AMP)

PyTorch's AMP handles everything automatically!

#### **Basic Usage**

```python
from torch.cuda.amp import autocast, GradScaler

# Create gradient scaler
scaler = GradScaler()

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward pass (automatically scales loss)
    scaler.scale(loss).backward()
    
    # Optimizer step (automatically unscales gradients)
    scaler.step(optimizer)
    
    # Update scaler
    scaler.update()
```

**That's it! Training is now ~2-3× faster with 2× less memory.**

#### **How It Works**

```
1. autocast():
   ═══════════════════════════════════════════════════════════
   - Wraps forward pass
   - Automatically uses FP16 for compatible operations
   - Keeps FP32 for operations that need it
   ═══════════════════════════════════════════════════════════
   
2. scaler.scale(loss):
   ═══════════════════════════════════════════════════════════
   - Multiplies loss by scale factor (typically 2^16 = 65536)
   - Prevents gradient underflow
   ═══════════════════════════════════════════════════════════
   
3. scaler.step(optimizer):
   ═══════════════════════════════════════════════════════════
   - Unscales gradients (divides by scale factor)
   - Checks for inf/nan
   - If OK: optimizer.step()
   - If not: skip update
   ═══════════════════════════════════════════════════════════
   
4. scaler.update():
   ═══════════════════════════════════════════════════════════
   - Adjusts scale factor dynamically
   - Increases if no overflow
   - Decreases if overflow detected
   ═══════════════════════════════════════════════════════════
```

#### **With Gradient Clipping**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    
    # Unscale BEFORE clipping (IMPORTANT!)
    scaler.unscale_(optimizer)
    
    # Now clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
```

**Why unscale first?**
```
Gradients are scaled up (× 65536)
If you clip without unscaling:
  max_norm=1.0 becomes effectively max_norm=65536
  → Clipping doesn't work!

Correct order:
  1. backward (gradients scaled)
  2. unscale (gradients normal)
  3. clip (at correct magnitude)
  4. step (optimizer uses unscaled gradients)
```

#### **Complete Example**

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )
    
    def forward(self, x):
        return self.layers(x)

# Setup
model = LargeModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scaler = GradScaler()

# Training
for epoch in range(10):
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        
        # Forward in FP16
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward with scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### BFloat16 (BF16)

**Better alternative to FP16 for newer GPUs:**

```
FP16:
  Sign: 1 bit
  Exponent: 5 bits  ← Small range
  Mantissa: 10 bits ← High precision

BF16:
  Sign: 1 bit
  Exponent: 8 bits  ← Same as FP32!
  Mantissa: 7 bits  ← Lower precision

Benefits of BF16:
  ✅ Same range as FP32 (no overflow/underflow issues)
  ✅ Doesn't need loss scaling
  ✅ Works for more models out of the box
  ❌ Less precision than FP16 (usually OK)
```

**Usage:**
```python
# PyTorch 1.10+
with autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)

# No scaler needed for BF16!
loss.backward()
optimizer.step()
```

### When to Use Mixed Precision

✅ **Always Use (Almost)**

**Modern GPUs (2020+):**
```
NVIDIA: V100, A100, RTX 30xx, RTX 40xx
AMD: MI100, MI250
→ Huge speedup, minimal downsides
```

**Any Large Model:**
```
Model > 100M parameters
→ 2× memory savings crucial
```

❌ **Be Careful With:**

**Tasks requiring high precision:**
```
- Some reinforcement learning
- Adversarial training
- Custom numerical operations
→ Test carefully, may need FP32
```

**Very small models:**
```
Model < 10M parameters
→ Overhead might outweigh benefits
```

### Performance Comparison

**Training Speed:**
```
Without AMP:  100 it/s
With AMP:     250 it/s  (2.5× faster!)
```

**Memory Usage:**
```
Without AMP:  16 GB
With AMP:     8 GB   (2× less memory)
```

**Accuracy:**
```
Without AMP:  95.2%
With AMP:     95.1%  (negligible difference)
```

---

## Combining Techniques

### The Ultimate Setup: Checkpointing + AMP + Accumulation + Clipping

```python
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

class OptimizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.TransformerEncoderLayer(512, 8)
        self.layer2 = nn.TransformerEncoderLayer(512, 8)
        self.layer3 = nn.TransformerEncoderLayer(512, 8)
    
    def forward(self, x):
        # Gradient checkpointing
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = checkpoint(self.layer3, x)
        return x

model = OptimizedModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()

# Training configuration
accumulation_steps = 8  # Simulate batch=256 with batch=32
max_grad_norm = 1.0

# Training loop
model.train()
optimizer.zero_grad()

for epoch in range(num_epochs):
    for i, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        
        # Forward with mixed precision
        with autocast():
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Update every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=max_grad_norm
            )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Zero gradients
            optimizer.zero_grad()
```

**Benefits of combining:**
```
✅ Gradient Checkpointing: 50-70% memory reduction
✅ Mixed Precision:         2× memory reduction, 2-3× speed
✅ Gradient Accumulation:   Can train with large effective batch
✅ Gradient Clipping:       Stable training

Combined effect:
  - Can train 4-8× larger models
  - 2-3× faster training
  - Stable convergence
```

### Memory Calculation

**Example: BERT-Large (340M params)**

```
Standard Training:
═══════════════════════════════════════════════════════════
Model weights:     340M × 4 bytes = 1.36 GB
Optimizer (Adam):  340M × 8 bytes = 2.72 GB
Gradients:         340M × 4 bytes = 1.36 GB
Activations:       ~6 GB (batch=16)
═══════════════════════════════════════════════════════════
Total: ~11.5 GB → Requires 16 GB GPU
═══════════════════════════════════════════════════════════

With All Optimizations:
═══════════════════════════════════════════════════════════
Model weights:     340M × 2 bytes = 0.68 GB  (FP16)
Optimizer (Adam):  340M × 4 bytes = 1.36 GB  (FP32 master)
Gradients:         340M × 2 bytes = 0.68 GB  (FP16)
Activations:       ~2 GB (checkpointing)
═══════════════════════════════════════════════════════════
Total: ~4.7 GB → Fits on 8 GB GPU!
═══════════════════════════════════════════════════════════

Batch size via accumulation: 16 × 4 = 64 (effective)
Training speed: 2.5× faster than standard
```

---

## Practical Guidelines

### Decision Tree

```
Training large model (>100M params)?
├─ Yes → Continue
└─ No → Standard training is fine

GPU memory tight?
├─ Yes → Enable Gradient Checkpointing
└─ No → Continue

Modern GPU (V100, A100, RTX 30xx+)?
├─ Yes → Enable Mixed Precision (AMP)
└─ No → Skip AMP

Want larger effective batch?
├─ Yes → Use Gradient Accumulation
└─ No → Continue

Training RNN/GAN/very deep net?
├─ Yes → Add Gradient Clipping
└─ No → Done!
```

### Recommended Configurations

#### **Configuration 1: Small Model (< 50M params)**
```python
# Standard training, no optimization needed
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

#### **Configuration 2: Medium Model (50-500M params)**
```python
# Mixed precision only
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### **Configuration 3: Large Model (500M+ params)**
```python
# Full optimization: Checkpointing + AMP + Accumulation + Clipping
from torch.cuda.amp import autocast, GradScaler

model.gradient_checkpointing_enable()  # If supported
scaler = GradScaler()
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Common Pitfalls

**1. Forgetting to unscale before clipping**
```python
# WRONG
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ← Clipping scaled grads!
scaler.step(optimizer)

# RIGHT
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # ← Unscale first!
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
```

**2. Using AMP with operations that don't support FP16**
```python
# Some operations don't work well in FP16
with autocast():
    x = torch.fft.fft(x)  # May have issues
    
# Solution: Cast to FP32 for specific operations
with autocast():
    x = x.float()  # FP32
    x = torch.fft.fft(x)
    x = x.half()  # Back to FP16
```

**3. Not calling scaler.update()**
```python
# WRONG
scaler.scale(loss).backward()
scaler.step(optimizer)
# Missing: scaler.update() ← Scale factor never adjusted!

# RIGHT
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()  # ← Don't forget!
```

---

## Performance Benchmarks

### Real-World Examples

**BERT-Base Training:**
```
Configuration:        Standard  |  Optimized
────────────────────────────────────────────
GPU Memory:           16 GB     |  8 GB
Batch Size:           16        |  32 (effective: 128 via accumulation)
Time per Epoch:       120 min   |  50 min (2.4× faster)
Final Accuracy:       85.2%     |  85.3% (same)
────────────────────────────────────────────
Techniques: AMP + Checkpointing + Accumulation
```

**GPT-2 Medium Training:**
```
Configuration:        Standard  |  Optimized
────────────────────────────────────────────
GPU Memory:           24 GB     |  10 GB
Batch Size:           8         |  32
Samples/sec:          12        |  35 (2.9× faster)
Perplexity:           28.5      |  28.3 (slightly better)
────────────────────────────────────────────
Techniques: AMP + Gradient Checkpointing
```

**Vision Transformer (ViT-Large):**
```
Configuration:        Standard  |  Optimized
────────────────────────────────────────────
GPU Memory:           20 GB     |  8 GB
Batch Size:           32        |  64
Training Time:        8 hours   |  3.5 hours (2.3× faster)
Accuracy:             82.1%     |  82.4% (slightly better)
────────────────────────────────────────────
Techniques: AMP + Gradient Checkpointing
```

---

## Key Takeaways

### Essential Patterns

**1. Quick Speedup (Minimal Changes):**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Just wrap forward pass
with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

Result: 2-3× faster, 2× less memory
```

**2. Maximum Memory Savings:**
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
with autocast():
    output = model(data)

Result: 50-70% memory reduction
```

**3. Production Training Setup:**
```python
model.gradient_checkpointing_enable()
scaler = GradScaler()
accumulation_steps = 8

for i, (data, target) in enumerate(dataloader):
    with autocast():
        loss = criterion(model(data), target) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Quick Reference

| Technique | Memory Savings | Speed Improvement | When to Use |
|-----------|----------------|-------------------|-------------|
| **Mixed Precision** | 2× | 2-3× faster | Always (modern GPUs) |
| **Gradient Checkpointing** | 50-70% | 20-30% slower | Large models |
| **Gradient Accumulation** | None | Slight slower | Memory limited |
| **All Combined** | 4-8× | 2× faster | Production training |

### Final Recommendations

1. **Always use Mixed Precision** (if GPU supports it)
2. **Enable Checkpointing** for models >100M params
3. **Use Accumulation** when GPU memory is limited
4. **Add Clipping** for RNNs and GANs
5. **Combine techniques** for best results

```python
# The ultimate training loop (memorize this!)
from torch.cuda.amp import autocast, GradScaler

model.gradient_checkpointing_enable()
scaler = GradScaler()
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    with autocast():
        loss = criterion(model(data), target) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

*See also: [Gradient Stabilization](./gradient_stabilization.md) for more on clipping and accumulation*