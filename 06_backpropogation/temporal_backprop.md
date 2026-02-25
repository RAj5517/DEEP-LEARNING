# Temporal Backpropagation: BPTT and Variants

> **Complete guide to backpropagation through time for sequential models**

---

## Table of Contents

1. [What is Temporal Backpropagation?](#what-is-temporal-backpropagation)
2. [Why Sequences are Different](#why-sequences-are-different)
3. [BPTT (Full) - Backpropagation Through Time](#bptt-full)
4. [Truncated BPTT](#truncated-bptt)
5. [Problems and Solutions](#problems-and-solutions)
6. [Practical Implementation](#practical-implementation)
7. [When to Use What](#when-to-use-what)

---

## What is Temporal Backpropagation?

### The Challenge

Sequential data (text, speech, time series) has **temporal dependencies**:
- Output at time t depends on inputs from time t, t-1, t-2, ...
- Network must maintain **memory** across time steps
- Standard backprop doesn't handle this naturally

### The Solution

**Backpropagation Through Time (BPTT)**: Unroll the recurrent network across time steps and apply standard backpropagation to the unrolled structure.

```
RNN (Folded View):          RNN (Unrolled View):
                            
  ┌───┐                      ┌───┐   ┌───┐   ┌───┐   ┌───┐
  │ h │ ◄──┐                 │ h₀│──▶│ h₁│──▶│ h₂│──▶│ h₃│
  └─▲─┘    │                 └─▲─┘   └─▲─┘   └─▲─┘   └─▲─┘
    │      │                   │       │       │       │
    │   recurrent              │       │       │       │
  ┌─┴─┐    │                 ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
  │ x │────┘                 │ x₀│   │ x₁│   │ x₂│   │ x₃│
  └───┘                      └───┘   └───┘   └───┘   └───┘
  
Same network,                t=0     t=1     t=2     t=3
applied repeatedly           (Now can apply standard backprop!)
```

---

## Why Sequences are Different

### Spatial vs. Temporal

**Standard Feedforward (Spatial):**
```
Input → Layer₁ → Layer₂ → Layer₃ → Output
  x       h₁       h₂       h₃        y

Each layer is independent.
```

**Recurrent (Temporal):**
```
x₀ → [RNN] → h₀ → [RNN] → h₁ → [RNN] → h₂ → ...
      ↑_______|      ↑_______|      ↑_______|
      Same weights W  Same weights W  Same weights W
      
Each step depends on previous step!
```

### Weight Sharing Across Time

**Key Insight:** The same weight matrix W is used at every time step.

```
hₜ = tanh(W_hh·hₜ₋₁ + W_xh·xₜ + b)

Where:
  W_hh = hidden-to-hidden weights (recurrent)
  W_xh = input-to-hidden weights
  Same W_hh used at t=0, t=1, t=2, ...
```

**Gradient Computation:**
```
∂L/∂W_hh = Σₜ ∂Lₜ/∂W_hh  (sum across all time steps)
```

---

## BPTT (Full) - Backpropagation Through Time

### The Algorithm

**Forward Pass (Unroll Through Time):**

```
For t = 0 to T:
    hₜ = tanh(W_hh @ hₜ₋₁ + W_xh @ xₜ + b_h)
    yₜ = W_hy @ hₜ + b_y
    Lₜ = loss(yₜ, targetₜ)

Total Loss: L = Σₜ Lₜ
```

**Backward Pass (Backpropagate Through Time):**

```
For t = T down to 0:
    ∂L/∂yₜ = loss_gradient(yₜ, targetₜ)
    ∂L/∂hₜ = ∂L/∂yₜ @ W_hy.T + ∂L/∂hₜ₊₁  ← Gradient from future
    ∂L/∂W_hh += ∂L/∂hₜ × tanh'(zₜ) @ hₜ₋₁.T
    ∂L/∂W_xh += ∂L/∂hₜ × tanh'(zₜ) @ xₜ.T
```

### Visual Representation

```
Forward Pass:
════════════════════════════════════════════════════════════
     x₀        x₁        x₂        x₃
     ▼         ▼         ▼         ▼
    ┌─┐  →   ┌─┐  →   ┌─┐  →   ┌─┐
h₀  │·│      │·│      │·│      │·│  h₃
    └┬┘      └┬┘      └┬┘      └┬┘
     ▼        ▼        ▼        ▼
     y₀       y₁       y₂       y₃
     ▼        ▼        ▼        ▼
     L₀       L₁       L₂       L₃
════════════════════════════════════════════════════════════

Backward Pass (Gradient Flow):
════════════════════════════════════════════════════════════
                                    ∂L/∂L = 1
                                      │
     ∂L/∂h₀ ←  ∂L/∂h₁ ←  ∂L/∂h₂ ←  ∂L/∂h₃
       ▲         ▲         ▲         ▲
       │         │         │         │
     ∂L/∂y₀   ∂L/∂y₁   ∂L/∂y₂   ∂L/∂y₃
       ▲         ▲         ▲         ▲
       │         │         │         │
      L₀        L₁        L₂        L₃
════════════════════════════════════════════════════════════

Gradient accumulation:
∂L/∂W_hh = ∂L₀/∂W_hh + ∂L₁/∂W_hh + ∂L₂/∂W_hh + ∂L₃/∂W_hh
           └────────────────┬────────────────┘
                Sum across time steps
```

### Detailed Mathematics

**Forward:**
```
z₀ = W_hh @ h₋₁ + W_xh @ x₀ + b_h
h₀ = tanh(z₀)
y₀ = W_hy @ h₀ + b_y
L₀ = loss(y₀, target₀)

z₁ = W_hh @ h₀ + W_xh @ x₁ + b_h
h₁ = tanh(z₁)
y₁ = W_hy @ h₁ + b_y
L₁ = loss(y₁, target₁)

...and so on
```

**Backward (example for t=2):**

```
Step 1: Gradient from output
∂L₂/∂y₂ = loss'(y₂, target₂)

Step 2: Gradient to hidden state
∂L₂/∂h₂ = ∂L₂/∂y₂ @ W_hy.T + ∂L₃/∂h₂  
          └──────┬─────┘   └────┬────┘
          local gradient   from future (t=3)

Step 3: Gradient through activation
∂L₂/∂z₂ = ∂L₂/∂h₂ ⊙ tanh'(z₂)
        = ∂L₂/∂h₂ ⊙ (1 - h₂²)

Step 4: Gradients to weights
∂L₂/∂W_hh = ∂L₂/∂z₂ @ h₁.T
∂L₂/∂W_xh = ∂L₂/∂z₂ @ x₂.T

Step 5: Gradient to previous hidden state
∂L₂/∂h₁ = W_hh.T @ ∂L₂/∂z₂  ← Flows to t=1
```

### Code Example (Manual Implementation)

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_hy = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)
        
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))
    
    def forward(self, x, h_prev=None):
        """
        x: [seq_len, batch_size, input_size]
        """
        seq_len, batch_size, _ = x.shape
        
        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size)
        else:
            h = h_prev
        
        outputs = []
        
        # Forward pass through time
        for t in range(seq_len):
            # h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
            z = h @ self.W_hh.T + x[t] @ self.W_xh.T + self.b_h
            h = torch.tanh(z)
            
            # y_t = W_hy @ h_t + b_y
            y = h @ self.W_hy.T + self.b_y
            
            outputs.append(y)
        
        # Stack outputs: [seq_len, batch_size, output_size]
        return torch.stack(outputs, dim=0), h

# Training with Full BPTT
model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass (entire sequence)
x = torch.randn(50, 32, 10)  # seq_len=50, batch=32, input=10
targets = torch.randint(0, 5, (50, 32))  # [seq_len, batch]

outputs, final_hidden = model(x)

# Compute loss (sum over time)
loss = 0
for t in range(50):
    loss += criterion(outputs[t], targets[t])
loss = loss / 50  # Average

# Backward pass (through all 50 steps)
optimizer.zero_grad()
loss.backward()  # BPTT happens here!
optimizer.step()
```

### Advantages of Full BPTT

✅ **Exact gradients**: Considers all temporal dependencies  
✅ **Theoretically optimal**: Full gradient information  
✅ **Simple conceptually**: Just unroll and apply backprop

### Disadvantages of Full BPTT

❌ **Memory intensive**: Must store activations for all time steps  
❌ **Slow**: O(T) time for sequence of length T  
❌ **Vanishing/exploding gradients**: Gets worse with longer sequences  
❌ **Impractical for long sequences**: Can't handle 1000+ steps

---

## Truncated BPTT

### The Problem with Full BPTT

```
Sequence length = 1000 steps
Each step stores activations
Total memory = 1000 × hidden_size × batch_size
               
For: hidden=512, batch=32, float32 (4 bytes)
Memory = 1000 × 512 × 32 × 4 = 65.5 MB per sample!

Plus: Vanishing gradients over 1000 steps
```

### The Solution: Truncated BPTT

**Idea:** Only backpropagate through a fixed number of time steps (k₁), while still carrying forward the hidden state.

```
Full BPTT:
═══════════════════════════════════════════════════════════
Forward:  h₀ → h₁ → h₂ → h₃ → h₄ → h₅ → h₆ → h₇ → h₈
Backward: ∂h₀← ∂h₁← ∂h₂← ∂h₃← ∂h₄← ∂h₅← ∂h₆← ∂h₇← ∂h₈
═══════════════════════════════════════════════════════════
Backprop through ALL 9 steps

Truncated BPTT (k₁=3):
═══════════════════════════════════════════════════════════
Chunk 1:
Forward:  h₀ → h₁ → h₂ → h₃
Backward:      ∂h₁← ∂h₂← ∂h₃
                └─────┬─────┘
            Backprop only 3 steps

Chunk 2:
Forward:  h₃ → h₄ → h₅ → h₆  (h₃ from previous chunk)
Backward:      ∂h₄← ∂h₅← ∂h₆
                └─────┬─────┘
            Backprop only 3 steps
═══════════════════════════════════════════════════════════
```

### Two Key Parameters

1. **k₁ (Backprop steps)**: How many steps to backpropagate
2. **k₂ (Forward steps)**: How many steps before computing gradients

**Common strategies:**

**Strategy 1: k₁ = k₂** (Most common)
```
Process sequence in chunks of size k₁
Backprop through each chunk
```

**Strategy 2: k₁ < k₂**
```
Take k₂ forward steps
Then backprop through last k₁ steps
```

### Visual Comparison

```
Sequence of 12 steps, k₁=4:

Full BPTT:
┌──────────────────────────────────────────────┐
│ Forward:  0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 →│→ 9 → 10 → 11
│ Backward: 0 ← 1 ← 2 ← 3 ← 4 ← 5 ← 6 ← 7 ← 8 ←│← 9 ← 10 ← 11
└──────────────────────────────────────────────┘
Memory: 12 steps, Vanishing gradient over 12 steps

Truncated BPTT (k₁=4):
Chunk 1:                Chunk 2:              Chunk 3:
┌─────────────┐        ┌─────────────┐      ┌─────────────┐
│ Fwd: 0→1→2→3│        │ Fwd: 4→5→6→7│      │ Fwd: 8→9→10→11
│ Bck:   1←2←3│        │ Bck:   5←6←7│      │ Bck:   9←10←11
└─────────────┘        └─────────────┘      └─────────────┘
     ↓                       ↓                     ↓
  update                  update               update
  
Memory: Only 4 steps, Vanishing gradient over 4 steps (better!)
```

### Implementation

```python
def truncated_bptt(model, sequence, targets, chunk_size, optimizer):
    """
    Truncated BPTT implementation
    
    Args:
        sequence: [seq_len, batch, input_size]
        targets: [seq_len, batch]
        chunk_size: k₁ (truncation length)
    """
    seq_len = sequence.shape[0]
    hidden = None  # Initial hidden state
    
    total_loss = 0
    num_chunks = 0
    
    # Process sequence in chunks
    for i in range(0, seq_len, chunk_size):
        # Get chunk
        chunk_end = min(i + chunk_size, seq_len)
        x_chunk = sequence[i:chunk_end]
        y_chunk = targets[i:chunk_end]
        
        # Forward pass through chunk
        optimizer.zero_grad()
        
        if hidden is not None:
            hidden = hidden.detach()  # ← Detach from previous chunk!
        
        outputs, hidden = model(x_chunk, hidden)
        
        # Compute loss for chunk
        loss = 0
        for t in range(len(x_chunk)):
            loss += criterion(outputs[t], y_chunk[t])
        loss = loss / len(x_chunk)
        
        # Backward pass (only through this chunk!)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        num_chunks += 1
    
    return total_loss / num_chunks

# Usage
model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Long sequence
long_sequence = torch.randn(1000, 32, 10)  # 1000 time steps!
long_targets = torch.randint(0, 5, (1000, 32))

# Train with truncated BPTT (chunk_size=35)
avg_loss = truncated_bptt(
    model, long_sequence, long_targets, 
    chunk_size=35,  # Only backprop through 35 steps at a time
    optimizer=optimizer
)
```

### Key Implementation Detail: `.detach()`

```python
# CRITICAL: Detach hidden state between chunks
if hidden is not None:
    hidden = hidden.detach()  # ← Breaks gradient flow!
```

**Why detach?**
```
Without detach:
Chunk 1: h₀ → h₁ → h₂ → h₃
Chunk 2: h₃ → h₄ → h₅ → h₆
         ↑
    Gradient flows back to h₃, h₂, h₁, h₀
    (defeats the purpose of truncation!)

With detach:
Chunk 1: h₀ → h₁ → h₂ → h₃
Chunk 2: h₃ →┊h₄ → h₅ → h₆
         ↑   ┊
    Detach here (gradient stops)
    Only backprop through current chunk
```

### PyTorch Built-in (Using RNN Layers)

```python
import torch.nn as nn

class TruncatedBPTTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

model = TruncatedBPTTModel(10, 20, 5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop with truncated BPTT
sequence = torch.randn(1000, 32, 10)
targets = torch.randint(0, 5, (1000, 32))

chunk_size = 50
hidden = None

for i in range(0, 1000, chunk_size):
    # Get chunk
    x_chunk = sequence[i:i+chunk_size]
    y_chunk = targets[i:i+chunk_size]
    
    # Detach hidden state (crucial!)
    if hidden is not None:
        hidden = hidden.detach()
    
    # Forward
    optimizer.zero_grad()
    output, hidden = model(x_chunk, hidden)
    
    # Loss and backward
    loss = criterion(output.view(-1, 5), y_chunk.view(-1))
    loss.backward()
    
    # Clip gradients (important for RNNs!)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

### Advantages of Truncated BPTT

✅ **Memory efficient**: Only store k₁ steps, not entire sequence  
✅ **Practical for long sequences**: Can handle 1000+ steps  
✅ **Reduces vanishing gradient**: Shorter gradient paths  
✅ **Faster**: Less computation per update  
✅ **Still captures dependencies**: Hidden state carries information forward

### Disadvantages of Truncated BPTT

❌ **Approximate gradients**: Doesn't see full temporal dependencies  
❌ **Hyperparameter k₁**: Need to tune truncation length  
❌ **Bias**: May miss long-range patterns  
❌ **Still has gradient issues**: Just less severe than full BPTT

### Choosing k₁ (Truncation Length)

**Guidelines:**

```
Too small (k₁=5-10):
  ✅ Very memory efficient
  ✅ Very fast
  ❌ May miss important dependencies
  ❌ Training might be unstable

Sweet spot (k₁=20-50):
  ✅ Good balance
  ✅ Captures most important dependencies
  ✅ Manageable memory
  Use this as default!

Too large (k₁=100+):
  ✅ Captures long dependencies
  ❌ Memory intensive
  ❌ Vanishing gradients return
  ❌ Slow
```

**Rule of thumb:**
- **Character-level language modeling**: k₁ = 50-100
- **Word-level language modeling**: k₁ = 20-35
- **Speech recognition**: k₁ = 20-50
- **Time series**: k₁ = 50-200 (depends on sampling rate)

---

## Problems and Solutions

### Problem 1: Vanishing Gradients

**Why it happens:**

```
∂L/∂h₀ = ∂L/∂hₜ × (∂hₜ/∂hₜ₋₁) × ... × (∂h₁/∂h₀)
                   └──────────┬──────────┘
                   Product of T terms

If each term < 1:
  Product → 0 as T increases
  
Example with tanh:
  |tanh'(x)| ≤ 1
  If gradient = 0.9 at each step:
    After 10 steps: 0.9^10 = 0.35
    After 50 steps: 0.9^50 = 0.005  (vanished!)
```

**Solutions:**

✅ **LSTM/GRU**: Designed to mitigate vanishing gradients
```python
# Replace simple RNN with LSTM
self.rnn = nn.LSTM(input_size, hidden_size)
# LSTM has gates that control gradient flow
```

✅ **Gradient Clipping**: Prevents explosion (not vanishing)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

✅ **Better Initialization**: Xavier/He initialization
```python
nn.init.xavier_uniform_(self.W_hh)
```

✅ **Shorter Sequences**: Use truncated BPTT with smaller k₁

### Problem 2: Exploding Gradients

**Why it happens:**

```
If gradient > 1 at each step:
  Product → ∞ as T increases
  
Example:
  If gradient = 1.1 at each step:
    After 10 steps: 1.1^10 = 2.59
    After 50 steps: 1.1^50 = 117  (exploded!)
```

**Symptoms:**
- Loss becomes NaN
- Weights become very large
- Model outputs nonsense

**Solution: Gradient Clipping (ESSENTIAL)**

```python
# After loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Complete training loop
for x, y in dataloader:
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    # Clip gradients (CRITICAL for RNNs!)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

### Problem 3: Slow Training

**Causes:**
- Long sequences with full BPTT
- Large hidden size
- Deep RNN stacks

**Solutions:**

✅ **Truncated BPTT**: Reduce k₁
✅ **Batch Processing**: Process multiple sequences in parallel
✅ **Use LSTMs efficiently**: 
```python
# PyTorch LSTM is optimized
self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2)
# Much faster than manual implementation
```
✅ **Consider Transformers**: For tasks where position doesn't matter much

---

## Practical Implementation

### Complete Example: Character-Level Language Model

```python
import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM (better than simple RNN)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed_size]
        
        # LSTM forward
        output, hidden = self.lstm(x, hidden)
        # output: [batch, seq_len, hidden_size]
        
        # Decode
        output = self.fc(output)
        # output: [batch, seq_len, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        # LSTM has two hidden states (h and c)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h, c)

# Training function with Truncated BPTT
def train_epoch(model, dataloader, criterion, optimizer, chunk_size=50):
    model.train()
    total_loss = 0
    hidden = None
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        # data: [batch, seq_len]
        batch_size, seq_len = data.shape
        
        # Process in chunks
        for i in range(0, seq_len, chunk_size):
            # Get chunk
            chunk_end = min(i + chunk_size, seq_len)
            data_chunk = data[:, i:chunk_end]
            target_chunk = targets[:, i:chunk_end]
            
            # Detach hidden state
            if hidden is not None:
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = model.init_hidden(batch_size)
            
            # Forward pass
            optimizer.zero_grad()
            output, hidden = model(data_chunk, hidden)
            
            # Reshape for loss computation
            output = output.reshape(-1, output.size(-1))
            target_chunk = target_chunk.reshape(-1)
            
            # Compute loss
            loss = criterion(output, target_chunk)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (IMPORTANT!)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Setup
vocab_size = 100
model = CharRNN(
    vocab_size=vocab_size,
    embed_size=128,
    hidden_size=256,
    num_layers=2
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    loss = train_epoch(
        model, train_loader, criterion, optimizer,
        chunk_size=50  # Truncated BPTT with k₁=50
    )
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

## When to Use What

### Decision Tree

```
Do you have sequential data?
├─ No  → Use standard backprop (CNN, Feedforward)
└─ Yes → Continue...

Is sequence length < 50?
├─ Yes → Use Full BPTT (simple, exact gradients)
└─ No  → Continue...

Is sequence length < 500?
├─ Yes → Use Truncated BPTT (k₁=20-50)
└─ No  → Continue...

Is order/position critical?
├─ Yes → Use Truncated BPTT with LSTM (k₁=50-100)
└─ No  → Consider Transformer (attention-based)
```

### Recommendations by Task

**Character-Level Language Modeling:**
- Sequence length: 100-1000+
- **Recommendation**: Truncated BPTT, k₁=50-100, LSTM

**Word-Level Language Modeling:**
- Sequence length: 20-100
- **Recommendation**: Truncated BPTT, k₁=20-35, LSTM or Transformer

**Speech Recognition:**
- Sequence length: 100-1000+
- **Recommendation**: Truncated BPTT, k₁=20-50, LSTM/GRU or Transformer

**Time Series Prediction:**
- Sequence length: varies
- **Recommendation**: Truncated BPTT, k₁ = prediction horizon, LSTM

**Video Processing:**
- Sequence length: 100-1000+ frames
- **Recommendation**: Truncated BPTT, k₁=16-32, 3D CNN or Transformer

---

## Key Takeaways

1. **BPTT = Standard backprop on unrolled RNN**
2. **Full BPTT**: Exact but impractical for long sequences
3. **Truncated BPTT**: Practical solution, chunk sequences
4. **Always detach hidden states** between chunks
5. **Always clip gradients** for RNNs
6. **LSTM/GRU > Simple RNN** for gradient stability
7. **Choose k₁ based on task**: 20-50 is usually good

### Essential Code Pattern

```python
# Truncated BPTT pattern (memorize this!)
chunk_size = 50
hidden = None

for i in range(0, seq_len, chunk_size):
    # 1. Detach hidden
    if hidden is not None:
        hidden = hidden.detach()
    
    # 2. Forward
    optimizer.zero_grad()
    output, hidden = model(x_chunk, hidden)
    loss = criterion(output, target_chunk)
    
    # 3. Backward
    loss.backward()
    
    # 4. Clip gradients (ESSENTIAL!)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 5. Update
    optimizer.step()
```

---

*See also: [Through Attention](./through_attention.md) for modern alternative to BPTT*