# LSTM — Long Short-Term Memory (1997)

Hochreiter & Schmidhuber — The architecture that made RNNs practical for real sequential tasks.

---

## The Problem It Solved

Vanilla RNNs suffer from **vanishing gradients through time**. During backpropagation, gradients are multiplied by the weight matrix `Wₕ` at every timestep. For a sequence of length 100, this is 100 multiplications. If the largest eigenvalue of `Wₕ` is < 1, gradients shrink to zero in early layers. If > 1, they explode.

The practical result: vanilla RNNs can only "remember" ~10-20 timesteps back. Anything earlier is effectively forgotten. This makes them useless for tasks requiring long-range understanding — a sentence where the verb must agree with the subject 50 words back, or a time series where a pattern from 200 steps ago affects the current prediction.

**LSTM's solution:** Replace the single hidden state with two separate streams — a **cell state** `C` (long-term memory) and a **hidden state** `h` (working memory) — and use **learnable gates** to control what flows through each.

---

## The Core Mechanism: Gates

Everything in LSTM flows through sigmoid or tanh gates. A sigmoid gate outputs values in (0, 1):
- Output ≈ 0: gate is closed, nothing passes
- Output ≈ 1: gate is open, everything passes
- Values in between: partial flow

This gating is **learnable** — the network learns when to open/close each gate based on the input and context.

### The Three Gates

**Forget Gate** — what to erase from long-term memory:
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```
Output ≈ 0: forget this cell state value completely
Output ≈ 1: keep this cell state value exactly

**Input Gate** — what new information to write to memory:
```
iₜ = σ(Wᵢ · [hₜ₋₁, xₜ] + bᵢ)    ← how much to write
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  ← what candidate values to write
```

**Output Gate** — what to expose from memory as the hidden state:
```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
```

---

## The Full LSTM Update

```
1. Forget gate:  fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
2. Input gate:   iₜ = σ(Wᵢ · [hₜ₋₁, xₜ] + bᵢ)
3. Candidate:    C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)

4. Cell update:  Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
                       ↑ forget old   ↑ write new

5. Output gate:  oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
6. Hidden state: hₜ = oₜ ⊙ tanh(Cₜ)
```

Where `⊙` is element-wise multiplication.

---

## Why the Cell State Fixes Vanishing Gradients

The critical line is step 4:

```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

This is an **additive update** — the new cell state is the old cell state plus some new information. During backpropagation, the gradient flows through this addition:

```
∂Cₜ/∂Cₜ₋₁ = fₜ
```

The gradient is multiplied by the forget gate `fₜ`, not the full weight matrix. And crucially — if the forget gate learns to stay ≈ 1 (keep everything), the gradient flows back through time with minimal decay. The network can learn to maintain gradients for hundreds of timesteps by keeping the forget gate open.

Compare to vanilla RNN where the gradient is multiplied by `Wₕ` at every step — no control, always decays or explodes.

---

## Architecture Diagram

```
         ┌─────────────────────────────────────────────────────────┐
         │                       LSTM Cell                         │
         │                                                          │
Cₜ₋₁ ───┼──────────────────(×fₜ)──────(+)────────────────── Cₜ   │
         │                              ↑                           │
         │                         (iₜ × C̃ₜ)                       │
         │                                                          │
         │  hₜ₋₁ ─┬──→ [σ] fₜ (forget) ──→ controls Cₜ₋₁         │
xₜ  ─────┼────────┼──→ [σ] iₜ (input)  ─┐                         │
         │         ├──→ [tanh] C̃ₜ      ─┘→ what to write           │
         │         └──→ [σ] oₜ (output) ──→ hₜ = oₜ ⊙ tanh(Cₜ)  │
         │                                                   ↓      │
         └──────────────────────────────────────────── hₜ ──┘      
                                                        = output
```

Two outputs at each step:
- `hₜ` — the hidden state (short-term working memory, passed to next step AND used as output)
- `Cₜ` — the cell state (long-term memory, passed to next step only)

---

## LSTM in Practice

### PyTorch

```python
import torch
import torch.nn as nn

# Single LSTM layer
lstm = nn.LSTM(
    input_size=128,       # Input feature dimension
    hidden_size=256,      # Hidden state / cell state dimension
    num_layers=2,         # Stack 2 LSTM layers
    batch_first=True,     # Input shape: [batch, seq_len, input_size]
    dropout=0.3,          # Dropout between stacked layers (not last layer)
    bidirectional=False
)

# Input
x = torch.randn(32, 100, 128)    # [batch=32, seq_len=100, features=128]

# Initial states (zeros by default if not provided)
h0 = torch.zeros(2, 32, 256)    # [num_layers, batch, hidden_size]
c0 = torch.zeros(2, 32, 256)

# Forward pass
output, (hn, cn) = lstm(x, (h0, c0))
# output: [32, 100, 256]  — hidden state at every timestep
# hn:     [2, 32, 256]    — final hidden state (last timestep, all layers)
# cn:     [2, 32, 256]    — final cell state (last timestep, all layers)
```

### Accessing Outputs

```python
# For sequence classification: use final hidden state
final_hidden = hn[-1]                      # Last layer's final hidden: [batch, 256]
logits = classifier(final_hidden)

# For sequence labeling (NER, tagging): use all timestep outputs
logits = classifier(output)               # [batch, seq_len, num_classes]

# For encoder-decoder: pass (hn, cn) as decoder initial state
_, (encoder_h, encoder_c) = lstm(src)
output, _ = decoder_lstm(tgt, (encoder_h, encoder_c))
```

### Sequence Padding and Packing

Real sequences in a batch have different lengths. Use packing to avoid computing on padding tokens:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Pack before LSTM, unpack after
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
packed_output, (hn, cn) = lstm(packed)
output, _ = pad_packed_sequence(packed_output, batch_first=True)
```

---

## Dropout in LSTM

Dropout in RNNs is tricky — you **cannot** drop the recurrent connection `hₜ₋₁` with standard dropout. Dropping a different mask each step destroys information flow. PyTorch's `nn.LSTM(dropout=p)` applies dropout only to the **inputs and outputs between stacked layers**, not to recurrent connections.

**Variational dropout (correct approach):** Apply the same mask at every timestep within a sequence. `WeightDrop` from the `awd-lstm` codebase or explicit implementation:

```python
# Simple version: apply dropout to h output, same mask across all timesteps
# Libraries like torchnlp handle this automatically
```

For most practical purposes, PyTorch's built-in dropout is sufficient.

---

## When LSTM Outperforms GRU

- Tasks with **very long sequences** where the separate cell/hidden distinction matters
- When you need to control memory write/read/forget independently
- Time series with multiple interacting timescales (the cell state handles slow timescale, hidden state handles fast)

In practice: start with GRU (faster, simpler), switch to LSTM if performance is insufficient or sequence length is very long (>500 steps).

---

## Common Patterns

### Sentiment Classification (Many-to-One)
```python
output, (hn, _) = lstm(embedded_text)
prediction = classifier(hn[-1])    # Use final hidden state
```

### Named Entity Recognition (Many-to-Many, same length)
```python
output, _ = lstm(embedded_text)    # [batch, seq, hidden]
predictions = classifier(output)   # Classify each token
```

### Language Model (Many-to-Many, predict next token)
```python
output, _ = lstm(embedded_tokens)
next_token_logits = classifier(output)  # Predict next token at each position
```

### Time Series Forecasting
```python
output, (hn, _) = lstm(time_series)
future_values = regressor(hn[-1])   # Predict future from final state
```