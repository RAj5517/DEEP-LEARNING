# RNN — Recurrent Neural Network

---

## What Is It?

A Recurrent Neural Network is a neural network designed for **sequential data** — data where order matters and context from previous steps is needed to understand the current step. Text, time series, speech, video frames, financial data — anything where "what came before" changes the meaning of "what comes now."

The core idea: give the network **memory** by feeding its own previous output back as input at each step. Unlike an MLP that sees one fixed input, an RNN processes a sequence one element at a time and maintains a **hidden state** that accumulates context across all previous steps.

```
MLP:  x → [Network] → y               (no memory, one shot)
RNN:  x₁ → [RNN] → h₁ → y₁
       x₂ + h₁ → [RNN] → h₂ → y₂    (uses memory from previous step)
       x₃ + h₂ → [RNN] → h₃ → y₃
```

This makes RNNs fundamentally different from MLPs and CNNs: the same network weights are applied at **every timestep**, but the hidden state carries context forward. This is **weight sharing across time**, analogous to how CNNs share weights across space.

---

## Architecture

### The Recurrent Cell

At each timestep `t`, the cell takes:
- `xₜ` — current input
- `hₜ₋₁` — hidden state from previous step (the "memory")

And produces:
- `hₜ` — new hidden state (passed to next step)
- `yₜ` — output at this step (optional, depends on task)

```
Vanilla RNN update:
hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + b)
yₜ = Wᵧ · hₜ
```

### Unrolled Through Time

The same cell is reused at each timestep — "unrolling" shows how it looks:

```
              h₀ (zeros)
              │
x₁ →  [RNN Cell] → h₁ → y₁
              │
x₂ →  [RNN Cell] → h₂ → y₂
              │
x₃ →  [RNN Cell] → h₃ → y₃
              │
x₄ →  [RNN Cell] → h₄ → y₄
```

All cells share the same weights `Wₕ, Wₓ, Wᵧ`. Backpropagation through this unrolled graph is called **Backpropagation Through Time (BPTT)**.

### The Vanishing Gradient Problem

BPTT multiplies gradients at every timestep during backprop. For a sequence of length T:

```
∂L/∂h₀ = ∂L/∂hₜ · (Wₕ)ᵀ   repeated T times
```

If `|Wₕ| < 1`: gradients shrink exponentially → early steps learn nothing (vanishing)
If `|Wₕ| > 1`: gradients grow exponentially → training diverges (exploding)

**This is why vanilla RNNs can't learn long-range dependencies** — anything more than ~10 steps back is effectively forgotten. LSTM and GRU were designed specifically to fix this.

---

## Key Concepts

### Hidden State as Memory

The hidden state `hₜ` is the RNN's "working memory" — a fixed-size vector (e.g., 256 dimensions) that must compress all relevant past information into that limited space. Everything the network needs to know about the past must be encoded into `h`.

This is both the strength (compact memory) and weakness (limited capacity, hard to retain very old information) of RNNs.

### Many-to-Many / Many-to-One / One-to-Many

RNNs are flexible in input/output structure:

```
Many-to-One:    x₁x₂x₃x₄ → y              Sentiment classification
One-to-Many:    x → y₁y₂y₃y₄              Image captioning
Many-to-Many:   x₁x₂x₃ → y₁y₂y₃          POS tagging (same length)
Seq-to-Seq:     x₁x₂x₃ → y₁y₂y₄y₅        Translation (different length)
```

For different-length input/output sequences, the **Encoder-Decoder** architecture is used (see below).

### Stacked (Deep) RNNs

Stack multiple RNN layers — the hidden state of one layer becomes the input sequence for the next:

```python
rnn = nn.LSTM(input_size=128, hidden_size=256, 
              num_layers=3,        # 3 stacked LSTM layers
              dropout=0.3,         # Dropout between layers
              batch_first=True)
```

More layers → more capacity to model complex patterns, but harder to train.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Use LSTM — not vanilla RNN — in practice
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # Input: [batch, seq_len, features]
            dropout=0.3,
            bidirectional=True
        )
        
        # ×2 for bidirectional
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.dropout(self.embedding(x))           # [batch, seq, embed]
        output, (hidden, cell) = self.rnn(x)          # output: [batch, seq, hidden*2]
        
        # Use final hidden state for classification
        # hidden: [num_layers*2, batch, hidden] for bidirectional
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Last layer, both directions
        return self.classifier(self.dropout(hidden))
```

---

## Encoder-Decoder (Seq2Seq)

For tasks where input and output are **different-length sequences** — translation, summarization, speech-to-text.

```
Input:  "How are you?"     [3 tokens]
Output: "Wie geht es?"     [4 tokens]  (different length)
```

Standard RNN can't handle this — its output length is fixed to input length. Encoder-Decoder solves it by separating the two phases:

```
ENCODER: reads entire input sequence → compresses into context vector c

  "How" → [LSTM] → h₁
  "are" → [LSTM] → h₂
  "you" → [LSTM] → h₃ = c    ← context vector (final hidden state)

DECODER: generates output token-by-token, conditioned on c

  c + <START> → [LSTM] → "Wie"
  c + "Wie"   → [LSTM] → "geht"
  c + "geht"  → [LSTM] → "es"
  c + "es"    → [LSTM] → <END>
```

```python
# Encoder
encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
_, (context_h, context_c) = encoder(src_sequence)

# Decoder — step by step at inference
decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
hidden, cell = context_h, context_c

generated = []
current_token = start_token
for _ in range(max_length):
    out, (hidden, cell) = decoder(current_token, (hidden, cell))
    current_token = fc(out).argmax()          # Greedy decoding
    generated.append(current_token)
    if current_token == end_token:
        break
```

**The bottleneck problem:** The entire input must be compressed into a single fixed-size vector `c`. For long sequences, information is lost. A 100-word sentence must fit into a 512-dimensional vector — the decoder "forgets" early parts of long inputs.

---

## Attention — The Bridge to Transformers

Attention (Bahdanau et al., 2015) fixed the Encoder-Decoder bottleneck. Instead of one context vector, the decoder can **look at all encoder hidden states** at each decoding step and decide which input positions to focus on:

```
Without attention:  decoder only sees final encoder state c
With attention:     decoder sees all [h₁, h₂, ..., hₙ] with learned weights

Attention weight αₜᵢ = how much decoder step t should focus on encoder step i

context_t = Σᵢ αₜᵢ · hᵢ     (weighted sum of all encoder states)
```

The attention weights are learned — the network learns which input words to look at when generating each output word. For translation: when generating "chien" (dog), the decoder attends strongly to "dog" in the input.

**This mechanism — computing weighted combinations of value vectors based on query-key similarity — is the core of the Transformer**. The Transformer removed the RNN entirely and made attention the primary operation. Full treatment in `transformer.md`.

---

## RNN Family Tree

```
RNN Family
│
├── Vanilla RNN              → Baseline, vanishing gradient problem
│
├── LSTM (1997) ⭐           → Gates + cell state, long-range memory
│     └── Stacked LSTM       → Multiple LSTM layers
│
├── GRU (2014) ⭐            → Simplified LSTM, fewer parameters
│
├── Bidirectional            → Wraps LSTM or GRU to see both directions
│     ├── BiLSTM             → Text classification, NER, tagging
│     └── BiGRU              → Same, lighter weight
│
└── Encoder-Decoder (Seq2Seq)
      ├── RNN Encoder-Decoder   → Translation, summarization (pre-Transformer)
      └── + Attention (2015)    → Led directly to Transformer architecture
```

**Deep dives:**
- LSTM → `lstm.md`
- GRU → `gru.md`
- Bidirectional, Encoder-Decoder, Attention intro → `rnn_variants.md`

---

## RNN vs Transformer — When to Use Each

| Use Case | Choice | Reason |
|----------|--------|--------|
| Streaming / real-time sequence | LSTM/GRU | Processes step-by-step, no full sequence needed |
| Very long sequences (10k+ steps) | LSTM/GRU | Transformer attention is O(n²) |
| Time series on edge devices | GRU | Tiny, fast inference |
| Small dataset NLP | BiLSTM | Less data-hungry than Transformers |
| Modern NLP (full sequence available) | Transformer | Better at scale |
| Most new projects | Transformer | Better tooling, pretrained models, performance |