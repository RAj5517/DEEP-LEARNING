# GRU — Gated Recurrent Unit (2014)

Cho et al. — "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation."

---

## What Is It?

GRU is a streamlined version of LSTM. It emerged from the same group (Bengio lab) that developed the Encoder-Decoder architecture, as they needed a recurrent cell that was effective but computationally lighter for sequence-to-sequence tasks.

The design philosophy: **LSTM has three gates and two states — do you need all of that?** GRU's answer was no. It merges the forget and input gates into a single **update gate**, eliminates the separate cell state, and achieves nearly identical performance on most tasks with ~25% fewer parameters.

---

## The GRU Mechanism

GRU has two gates and one hidden state (no separate cell state):

### Reset Gate — how much to forget from the past

```
rₜ = σ(Wr · [hₜ₋₁, xₜ])
```

- `rₜ ≈ 0`: ignore the previous hidden state completely (reset memory)
- `rₜ ≈ 1`: use the previous hidden state fully

### Update Gate — how much to update vs retain

```
zₜ = σ(Wz · [hₜ₋₁, xₜ])
```

- `zₜ ≈ 0`: keep the previous hidden state (ignore new input)
- `zₜ ≈ 1`: replace with the new candidate state

### Candidate Hidden State

```
h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ])
```

The reset gate `rₜ` masks how much of `hₜ₋₁` is used to compute the candidate. If `rₜ ≈ 0`, the candidate is computed ignoring the past — fresh start.

### Final Hidden State

```
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

This is the key equation. The update gate `zₜ` directly **interpolates** between:
- Keeping the old hidden state `hₜ₋₁` (when `zₜ → 0`)
- Taking the new candidate `h̃ₜ` (when `zₜ → 1`)

When `zₜ ≈ 0` for many consecutive steps, the hidden state is preserved unchanged — this is how GRU maintains long-term memory. The update gate learns to "freeze" memory when the input isn't relevant.

---

## Full Update Equations

```
rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)          ← reset gate
zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)          ← update gate
h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ] + b)   ← candidate state
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ      ← new hidden state
```

Only one output: `hₜ` (the hidden state, same as output).

---

## Architecture Diagram

```
         ┌────────────────────────────────────────────────┐
         │                  GRU Cell                       │
         │                                                  │
hₜ₋₁ ───┼─────────┬──────────────────────────────────┐   │
         │         │                                   ↓   │
xₜ  ────┼─────────┼──→ [σ] → rₜ (reset)              │   │
         │         │                                   │   │
         │         ├──→ [σ] → zₜ (update) ────────────┤   │
         │         │                      ↓            │   │
         │         └──→ [tanh] with rₜ→ h̃ₜ           │   │
         │                                ↓            ↓   │
         │                    hₜ = (1-zₜ)·hₜ₋₁ + zₜ·h̃ₜ  │
         │                                ↓                │
         └────────────────────────── hₜ ─────────────────┘
                                      = output
```

Single output `hₜ` — no separate cell state unlike LSTM.

---

## GRU vs LSTM

| | LSTM | GRU |
|--|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| States | 2 (hidden + cell) | 1 (hidden only) |
| Parameters per unit | 4 × (input + hidden) × hidden | 3 × (input + hidden) × hidden |
| Total params ratio | ~100% | ~75% |
| Training speed | Slower | Faster (~25%) |
| Long-range memory | Slightly better | Slightly worse |
| Short sequences (<100) | Comparable | Comparable |
| Long sequences (>500) | Slight advantage | Slight disadvantage |

**Rule of thumb:** Start with GRU. Switch to LSTM if:
- Sequence length is very long (>500 steps)
- The task seems to need the separate cell state abstraction
- GRU is underfitting and LSTM is worth trying

---

## PyTorch Usage

```python
import torch
import torch.nn as nn

gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=False
)

x = torch.randn(32, 100, 128)    # [batch, seq_len, features]

output, hn = gru(x)
# output: [32, 100, 256]   — hidden state at every timestep
# hn:     [2, 32, 256]     — final hidden state (all layers)
# Note: no cell state cn — simpler API than LSTM
```

**Bidirectional GRU:**

```python
bigru = nn.GRU(
    input_size=128, hidden_size=256,
    num_layers=2, batch_first=True,
    bidirectional=True          # Output is hidden_size * 2 = 512
)

output, hn = bigru(x)
# output: [32, 100, 512]   — both directions concatenated
# hn:     [4, 32, 256]     — [num_layers*2, batch, hidden] for bidirectional

# Get last layer from both directions:
final = torch.cat([hn[-2], hn[-1]], dim=1)   # [32, 512]
```

---

## Common Applications

GRU is used in the same domains as LSTM. It tends to be preferred when:

**Time series forecasting:**
```python
class GRUForecaster(nn.Module):
    def __init__(self, input_features, hidden_size, forecast_horizon):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_size, num_layers=2,
                          batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden_size, forecast_horizon)
    
    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn[-1])   # Predict future from final state
```

**Text classification:**
```python
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=2,
                          batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        x = self.embed(x)
        output, hn = self.gru(x)
        # Mean pooling over sequence (often better than just final state)
        pooled = output.mean(dim=1)
        return self.classifier(pooled)
```

**Reinforcement Learning (Recurrent Policy):**
GRU is preferred over LSTM in RL (PPO, A2C with recurrent policies) because it's faster and the separate cell state rarely provides advantage in the short horizons typical of RL episodes.

---

## Why GRU Was Important

GRU appeared in the same paper that introduced the Encoder-Decoder architecture. The paper needed an RNN cell that was powerful enough to encode variable-length sentences but light enough to train efficiently. GRU was designed for this.

It proved that LSTM's complexity wasn't strictly necessary — a simpler gating mechanism could achieve nearly the same result. This line of thinking (simplify the architecture, keep the performance) eventually influenced the Transformer, which eliminated recurrence entirely and achieved far better results with pure attention.

GRU sits at the midpoint: simpler than LSTM, more capable than vanilla RNN. For production systems where inference speed matters (real-time speech, on-device NLP, RL), GRU remains the practical choice over both.