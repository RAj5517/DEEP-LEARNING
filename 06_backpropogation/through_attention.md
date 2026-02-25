# Backpropagation Through Attention

> **Complete guide to gradient flow in Transformer attention mechanisms**

---

## Table of Contents

1. [What is Attention?](#what-is-attention)
2. [Why Attention is Different](#why-attention-is-different)
3. [Self-Attention Forward Pass](#self-attention-forward-pass)
4. [Self-Attention Backward Pass](#self-attention-backward-pass)
5. [Multi-Head Attention](#multi-head-attention)
6. [Cross-Attention](#cross-attention)
7. [Practical Implementation](#practical-implementation)
8. [Advantages Over BPTT](#advantages-over-bptt)

---

## What is Attention?

### The Core Idea

**Attention** allows a model to focus on different parts of the input when producing each output. Unlike RNNs that process sequentially, attention can look at all positions **simultaneously**.

```
Traditional RNN:
═══════════════════════════════════════════════════════
x₀ → h₀ → x₁ → h₁ → x₂ → h₂ → x₃ → h₃
     Sequential processing

Attention:
═══════════════════════════════════════════════════════
x₀ ↘
x₁ → [Attention] → output
x₂ ↗
x₃ ↗
     All inputs considered at once!
```

### The Attention Equation

The fundamental attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
  Q = Queries   [seq_len, d_k]
  K = Keys      [seq_len, d_k]  
  V = Values    [seq_len, d_v]
  d_k = dimension of keys (for scaling)
```

**Intuition:**
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

**Similarity → Weights → Weighted Sum:**
```
1. Compute similarity: QK^T (dot product of queries and keys)
2. Scale: divide by √d_k (prevents softmax saturation)
3. Normalize: softmax (weights sum to 1)
4. Weighted sum: multiply by V
```

---

## Why Attention is Different

### Compared to RNNs

**RNN (Sequential):**
```
Pros:
  - Natural for sequences
  - Compact hidden state

Cons:
  - Sequential computation (slow)
  - Vanishing gradients over long sequences
  - Can't parallelize
```

**Attention (Parallel):**
```
Pros:
  - Parallel computation (fast)
  - Direct connections to all positions
  - No vanishing gradient problem
  - Can model long-range dependencies

Cons:
  - Quadratic memory O(n²)
  - No inherent position information
```

### Gradient Flow

**RNN:**
```
Gradients flow sequentially through time:
∂L/∂h₀ ← ∂L/∂h₁ ← ∂L/∂h₂ ← ... ← ∂L/∂hₙ

Problems:
  - Long path → vanishing gradients
  - Sequential → slow
```

**Attention:**
```
Gradients flow directly through attention weights:
∂L/∂x₀ ← [Attention] ← ∂L/∂output
∂L/∂x₁ ←  Weights   ←
∂L/∂x₂ ←            ←

Benefits:
  - Short path → stable gradients
  - Parallel → fast
```

---

## Self-Attention Forward Pass

### The Architecture

```
Input Sequence X: [seq_len, d_model]
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    [W_Q]       [W_K]       [W_V]
        │           │           │
        ▼           ▼           ▼
      Q           K           V
   [seq,d_k]  [seq,d_k]  [seq,d_v]
        │           │           │
        └─────┬─────┴───────────┘
              │
              ▼
    Attention(Q, K, V)
              │
              ▼
         Output Z
       [seq, d_v]
```

### Step-by-Step Forward Pass

#### **Step 1: Linear Projections**

Create Q, K, V from input X:

```python
Q = X @ W_Q  # [seq_len, d_k]
K = X @ W_K  # [seq_len, d_k]
V = X @ W_V  # [seq_len, d_v]

# W_Q, W_K, W_V are learnable weight matrices
# Usually d_k = d_v = d_model / num_heads
```

**Example:**
```
Input X: [4, 512]  (4 tokens, 512 dimensions)
W_Q: [512, 64]     (project to 64 dimensions)

Q = X @ W_Q = [4, 64]
K = X @ W_K = [4, 64]
V = X @ W_V = [4, 64]
```

#### **Step 2: Compute Attention Scores**

Calculate similarity between all queries and keys:

```python
scores = Q @ K.T  # [seq_len, seq_len]
scores = scores / sqrt(d_k)  # Scale by √d_k
```

**Scaling is crucial:**
```
Without scaling:
  Dot products can be very large
  Softmax saturates (gradient → 0)
  
With scaling (√d_k):
  Keeps scores in reasonable range
  Softmax gradients remain healthy
```

**Visualization:**
```
Q: [4, 64]  K^T: [64, 4]

scores = Q @ K^T = [4, 4]

       k₀   k₁   k₂   k₃
    ┌────┬────┬────┬────┐
 q₀ │ s₀₀│ s₀₁│ s₀₂│ s₀₃│  How similar is query 0 to each key?
    ├────┼────┼────┼────┤
 q₁ │ s₁₀│ s₁₁│ s₁₂│ s₁₃│  How similar is query 1 to each key?
    ├────┼────┼────┼────┤
 q₂ │ s₂₀│ s₂₁│ s₂₂│ s₂₃│
    ├────┼────┼────┼────┤
 q₃ │ s₃₀│ s₃₁│ s₃₂│ s₃₃│
    └────┴────┴────┴────┘
    
Each row: how much token i attends to all other tokens
```

#### **Step 3: Apply Softmax**

Convert scores to attention weights (probabilities):

```python
attention_weights = softmax(scores, dim=-1)  # [seq_len, seq_len]
```

**Properties:**
- Each row sums to 1.0
- All values in [0, 1]
- Higher scores → higher weights

```
After softmax:

       k₀    k₁    k₂    k₃     ← Keys (what to attend to)
    ┌─────┬─────┬─────┬─────┐
 q₀ │ 0.1 │ 0.6 │ 0.2 │ 0.1 │  Token 0 attends mostly to token 1
    ├─────┼─────┼─────┼─────┤
 q₁ │ 0.3 │ 0.3 │ 0.3 │ 0.1 │  Token 1 spreads attention evenly
    ├─────┼─────┼─────┼─────┤
 q₂ │ 0.1 │ 0.1 │ 0.7 │ 0.1 │  Token 2 attends mostly to itself
    ├─────┼─────┼─────┼─────┤
 q₃ │ 0.2 │ 0.2 │ 0.2 │ 0.4 │  Token 3 attends mostly to itself
    └─────┴─────┴─────┴─────┘
↑ Queries (what is attending)

Each row sums to 1.0
```

#### **Step 4: Apply Attention to Values**

Weighted sum of values:

```python
output = attention_weights @ V  # [seq_len, d_v]
```

**What this does:**
```
For each query position:
  output[i] = Σⱼ attention_weights[i,j] * V[j]
  
Example for position 0:
  output[0] = 0.1*V[0] + 0.6*V[1] + 0.2*V[2] + 0.1*V[3]
              └──────────────────┬───────────────────┘
              Weighted combination based on attention
```

### Complete Forward Pass Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def self_attention_forward(X, W_Q, W_K, W_V, mask=None):
    """
    Self-attention forward pass
    
    Args:
        X: [batch, seq_len, d_model] - input
        W_Q, W_K, W_V: [d_model, d_k] - weight matrices
        mask: [batch, seq_len, seq_len] - optional mask
    
    Returns:
        output: [batch, seq_len, d_v]
        attention_weights: [batch, seq_len, seq_len]
    """
    batch_size, seq_len, d_model = X.shape
    
    # Step 1: Linear projections
    Q = X @ W_Q  # [batch, seq_len, d_k]
    K = X @ W_K  # [batch, seq_len, d_k]
    V = X @ W_V  # [batch, seq_len, d_v]
    
    d_k = Q.shape[-1]
    
    # Step 2: Compute attention scores
    scores = Q @ K.transpose(-2, -1)  # [batch, seq_len, seq_len]
    scores = scores / math.sqrt(d_k)  # Scale
    
    # Optional: Apply mask (for causal attention or padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 3: Softmax
    attention_weights = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]
    
    # Step 4: Apply attention to values
    output = attention_weights @ V  # [batch, seq_len, d_v]
    
    return output, attention_weights
```

---

## Self-Attention Backward Pass

Now for the gradient computation! This is where PyTorch's autograd shines, but understanding the flow is crucial.

### Computational Graph

```
         X (input)
         │
    ┌────┼────┐
    │    │    │
  W_Q  W_K  W_V
    │    │    │
    Q    K    V
    │    │    │
    └─┬──┘    │
      │       │
   scores     │
      │       │
   softmax    │
      │       │
 attention_wts│
      │       │
      └───┬───┘
          │
       output
          │
        loss
```

### Gradient Flow (Backward)

#### **Step 4 Backward: Through Attention Application**

```
Forward:  output = attention_weights @ V
Backward: ∂L/∂V = attention_weights.T @ ∂L/∂output
         ∂L/∂attention_weights = ∂L/∂output @ V.T
```

**Mathematics:**
```
output[i] = Σⱼ attention_weights[i,j] * V[j]

∂L/∂V[j] = Σᵢ attention_weights[i,j] * ∂L/∂output[i]

∂L/∂attention_weights[i,j] = ∂L/∂output[i]^T @ V[j]
```

#### **Step 3 Backward: Through Softmax**

Softmax gradient is tricky:

```
Forward:  p = softmax(scores)
Backward: ∂L/∂scores = p ⊙ (∂L/∂p - Σⱼ p ⊙ ∂L/∂p)

Where ⊙ is element-wise multiplication
```

**Why it's complex:**
- Softmax couples all elements in a row
- Gradient depends on all outputs, not just one

**Intuition:**
```
If p[i] goes up:
  - All other p[j] must go down (they sum to 1)
  - Gradient reflects this coupling
```

#### **Step 2 Backward: Through Score Computation**

```
Forward:  scores = (Q @ K.T) / √d_k
Backward: ∂L/∂Q = (∂L/∂scores @ K) / √d_k
         ∂L/∂K = (∂L/∂scores.T @ Q) / √d_k
```

#### **Step 1 Backward: Through Linear Projections**

```
Forward:  Q = X @ W_Q
Backward: ∂L/∂X += ∂L/∂Q @ W_Q.T
         ∂L/∂W_Q = X.T @ ∂L/∂Q

Similarly for K and V:
∂L/∂X += ∂L/∂K @ W_K.T
∂L/∂X += ∂L/∂V @ W_V.T
∂L/∂W_K = X.T @ ∂L/∂K
∂L/∂W_V = X.T @ ∂L/∂V
```

### Complete Backward Pass Visualization

```
                    ∂L/∂output
                         │
                         ▼
                    ┌─────────┐
                    │ output  │
                    │= attn@V │
                    └────┬────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    ┌──────────────┐          ┌─────────────┐
    │∂L/∂attention │          │   ∂L/∂V    │
    │   _weights   │          └──────┬──────┘
    └──────┬───────┘                 │
           │                         ▼
           │                  ┌─────────────┐
           ▼                  │  ∂L/∂W_V   │
    ┌──────────────┐          │ ∂L/∂X (V)  │
    │  ∂L/∂scores │          └─────────────┘
    │  (softmax')  │
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  / √d_k      │
    └──────┬───────┘
           │
     ┌─────┴──────┐
     │ = Q@K^T    │
     └──────┬─────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌────────┐    ┌────────┐
│ ∂L/∂Q  │    │ ∂L/∂K  │
└───┬────┘    └───┬────┘
    │             │
    ▼             ▼
┌────────┐    ┌────────┐
│∂L/∂W_Q │    │∂L/∂W_K │
│∂L/∂X(Q)│    │∂L/∂X(K)│
└────────┘    └────────┘
```

### Automatic Differentiation Example

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Parameter(torch.randn(d_model, d_k) * 0.01)
        self.W_K = nn.Parameter(torch.randn(d_model, d_k) * 0.01)
        self.W_V = nn.Parameter(torch.randn(d_model, d_k) * 0.01)
    
    def forward(self, X):
        # Forward pass
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ V
        
        return output

# Create model
model = SelfAttention(d_model=512, d_k=64)
X = torch.randn(2, 10, 512, requires_grad=True)  # [batch, seq, d_model]

# Forward pass
output = model(X)
loss = output.mean()  # Dummy loss

# Backward pass (automatic!)
loss.backward()

# Gradients are computed automatically:
print(f"∂L/∂W_Q: {model.W_Q.grad.shape}")  # [512, 64]
print(f"∂L/∂W_K: {model.W_K.grad.shape}")  # [512, 64]
print(f"∂L/∂W_V: {model.W_V.grad.shape}")  # [512, 64]
print(f"∂L/∂X: {X.grad.shape}")             # [2, 10, 512]
```

### Key Insight: No Vanishing Gradients!

**Why attention is better than RNN for long sequences:**

```
RNN:
∂L/∂h₀ = ∂L/∂hₙ × ∂hₙ/∂hₙ₋₁ × ... × ∂h₁/∂h₀
         └────────────┬─────────────┘
         Product of n terms → vanishes!

Attention:
∂L/∂x₀ = ∂L/∂output × ∂output/∂attention × ∂attention/∂scores × ∂scores/∂Q × ∂Q/∂x₀
         └──────────────────────┬──────────────────────┘
         Fixed depth, no multiplicative accumulation!
```

**Direct connections:**
- Every input position can directly influence every output position
- Gradient path length is constant (doesn't grow with sequence length)
- No vanishing gradient problem for long sequences!

---

## Multi-Head Attention

### Why Multiple Heads?

**Problem with single attention:**
- Single attention matrix has one "view" of relationships
- Might focus on syntax OR semantics, not both

**Solution:**
- Multiple attention heads
- Each head can specialize in different patterns
- Example: Head 1 → syntax, Head 2 → semantics, Head 3 → position

### Architecture

```
                    Input X
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     Head 1         Head 2         Head 3
    [Attn]         [Attn]         [Attn]
        │              │              │
        └──────────────┼──────────────┘
                       │
                   Concatenate
                       │
                       ▼
                  Linear (W_O)
                       │
                       ▼
                    Output
```

### Forward Pass

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projection matrices for all heads (combined)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, X, mask=None):
        batch_size, seq_len, d_model = X.shape
        
        # 1. Linear projections
        Q = self.W_Q(X)  # [batch, seq_len, d_model]
        K = self.W_K(X)
        V = self.W_V(X)
        
        # 2. Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for multi-head attention
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, d_k]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3. Apply attention for each head (in parallel!)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = attention_weights @ V
        # attention_output: [batch, num_heads, seq_len, d_k]
        
        # 4. Concatenate heads
        attention_output = attention_output.transpose(1, 2)
        # [batch, seq_len, num_heads, d_k]
        
        attention_output = attention_output.contiguous().view(
            batch_size, seq_len, d_model
        )
        # [batch, seq_len, d_model]
        
        # 5. Final linear projection
        output = self.W_O(attention_output)
        
        return output, attention_weights
```

### Backward Pass

**Good news:** PyTorch handles it automatically!

**What happens:**
1. Gradients flow back through W_O (output projection)
2. Split back into multiple heads
3. Each head computes gradients independently (parallel!)
4. Gradients combine through concatenation
5. Flow back through W_Q, W_K, W_V

```
Backward flow:
                    ∂L/∂output
                         │
                         ▼
                    ┌─────────┐
                    │   W_O   │
                    └────┬────┘
                         │
                    Concatenate
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    ∂L/∂head₁        ∂L/∂head₂        ∂L/∂head₃
    [Attention']     [Attention']     [Attention']
        │                │                │
        └────────────────┼────────────────┘
                         │
                    ∂L/∂X (accumulated)
```

---

## Cross-Attention

Used in encoder-decoder architectures (e.g., machine translation).

### The Difference

**Self-Attention:**
- Q, K, V all from the same sequence
- "I attend to myself"

**Cross-Attention:**
- Q from one sequence (e.g., decoder)
- K, V from another sequence (e.g., encoder)
- "I attend to something else"

### Architecture

```
Encoder Output      Decoder Hidden
      │                  │
      │ K, V             │ Q
      │                  │
      └──────┬───────────┘
             │
        [Attention]
             │
             ▼
         Output
```

### Example: Machine Translation

```
Source: "Hello world"  → Encoder → encoder_output
Target: "Bonjour"      → Decoder → decoder_hidden
                                       ↓ Q
                               K,V ← encoder_output
                                       ↓
                               Cross-Attention
```

### Code

```python
def cross_attention(decoder_hidden, encoder_output, W_Q, W_K, W_V):
    """
    Cross-attention: decoder attends to encoder
    
    Args:
        decoder_hidden: [batch, dec_len, d_model] (queries)
        encoder_output: [batch, enc_len, d_model] (keys, values)
    """
    # Queries from decoder
    Q = decoder_hidden @ W_Q  # [batch, dec_len, d_k]
    
    # Keys and Values from encoder
    K = encoder_output @ W_K  # [batch, enc_len, d_k]
    V = encoder_output @ W_V  # [batch, enc_len, d_v]
    
    # Attention mechanism (same as self-attention)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ V
    
    return output
```

### Backward Pass

Similar to self-attention, but gradients flow to two different sources:
- Gradients for Q → flow back to decoder
- Gradients for K, V → flow back to encoder

---

## Practical Implementation

### Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, mask)
        x = x + self.dropout1(attn_output)  # Residual
        x = self.norm1(x)                    # Layer norm
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)    # Residual
        x = self.norm2(x)                    # Layer norm
        
        return x

# Training
model = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Forward and backward
x = torch.randn(32, 10, 512)  # [batch, seq_len, d_model]
output = model(x)
loss = output.mean()  # Dummy loss

optimizer.zero_grad()
loss.backward()  # Gradients computed automatically!
optimizer.step()
```

### Key Implementation Details

**1. Residual Connections:**
```python
x = x + attention(x)  # ← Residual connection
```
- Helps gradient flow
- Prevents degradation in deep networks

**2. Layer Normalization:**
```python
x = layer_norm(x + attention(x))
```
- Stabilizes training
- Normalizes across feature dimension

**3. Dropout:**
```python
x = dropout(attention(x))
```
- Regularization
- Prevents overfitting

---

## Advantages Over BPTT

### 1. Parallel Computation

**RNN (Sequential):**
```
h₁ depends on h₀ → must compute sequentially
h₂ depends on h₁ → wait for h₁
h₃ depends on h₂ → wait for h₂
...
Cannot parallelize over sequence length!
```

**Attention (Parallel):**
```
All positions computed simultaneously
output[0], output[1], ..., output[n] in parallel
Huge speedup on GPUs!
```

### 2. Long-Range Dependencies

**RNN:**
```
Path from word 0 to word 100: 100 steps
Gradient flows through 100 multiplications
High chance of vanishing/exploding
```

**Attention:**
```
Path from word 0 to word 100: 1 step (direct connection)
Gradient flows through attention weights
No vanishing gradient problem!
```

### 3. Interpretability

**Attention weights are interpretable:**
```
Can visualize which words the model is focusing on
Example: "The cat sat on the mat"
  "sat" might attend strongly to "cat" (subject)
  "sat" might attend to "mat" (object)
```

### Disadvantages

❌ **Quadratic memory:** O(n²) for sequence length n  
❌ **No position information:** Need position encodings  
❌ **Inductive bias:** RNNs naturally handle sequential data  

---

## Key Takeaways

1. **Attention = Weighted sum based on similarity**
2. **Self-attention: Q, K, V from same sequence**
3. **Multi-head: Multiple attention patterns in parallel**
4. **Cross-attention: Attend to different sequence**
5. **No vanishing gradients:** Direct connections
6. **Parallel computation:** Much faster than RNNs
7. **PyTorch autograd handles backprop automatically**

### Essential Pattern

```python
# Self-attention pattern (memorize this!)
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = (Q @ K.T) / sqrt(d_k)
attention_weights = softmax(scores)
output = attention_weights @ V

# Backward happens automatically with loss.backward()
```

---

*See also: [Temporal Backprop](./temporal_backprop.md) for comparison with RNNs*