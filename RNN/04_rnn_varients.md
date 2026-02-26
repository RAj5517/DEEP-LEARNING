# RNN Variants — Vanilla RNN, Bidirectional, Encoder-Decoder, Attention

---

## Vanilla RNN

The original formulation. Every modern RNN variant (LSTM, GRU) is a solution to the problems vanilla RNN introduced.

```
hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + b)
yₜ = Wᵧ · hₜ
```

Two weight matrices: `Wₕ` (recurrent) and `Wₓ` (input). The same matrices are used at every timestep.

**The vanishing gradient problem in numbers:**
```
Gradient at step t=100 ∝ (Wₕ)^100

If max eigenvalue of Wₕ = 0.9:   0.9^100 ≈ 0.000027  (vanished)
If max eigenvalue of Wₕ = 1.1:   1.1^100 ≈ 13,780    (exploded)
```

The only stable regime is eigenvalue ≈ 1, which is hard to maintain during training as weights update. This is the fundamental reason vanilla RNN is not used in practice.

**Still useful to understand because:** LSTM and GRU are direct responses to exactly this failure mode. Understanding what broke makes the fixes (gates, cell state) intuitive.

**In PyTorch (for reference only — use LSTM/GRU in practice):**
```python
rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True)
output, hn = rnn(x)
```

---

## Bidirectional RNN

### The Idea

Standard RNNs are causal — at timestep `t`, they only have access to inputs `x₁, x₂, ..., xₜ`. For many tasks, future context is just as important as past context.

Example: in "The **bank** by the river was muddy," understanding that "bank" means riverbank (not financial institution) requires seeing "river" which comes **after** "bank" in the sentence.

Bidirectional RNN runs **two RNNs simultaneously** on the same sequence:
- Forward RNN: processes left → right
- Backward RNN: processes right → left

Their hidden states are concatenated at each timestep:

```
Forward:  x₁ → x₂ → x₃ → x₄ → x₅
          h̄₁   h̄₂   h̄₃   h̄₄   h̄₅

Backward: x₁ ← x₂ ← x₃ ← x₄ ← x₅
          h←₁  h←₂  h←₃  h←₄  h←₅

Output at each step: [h̄ₜ ; h←ₜ]  ← concatenation, 2× hidden size
```

### PyTorch

```python
bilstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True      # This single flag does everything
)

output, (hn, cn) = bilstm(x)
# output: [batch, seq_len, 512]   — 256 forward + 256 backward
# hn:     [4, batch, 256]         — [num_layers*2, batch, hidden]
#         hn[0]: layer 1 forward, hn[1]: layer 1 backward
#         hn[2]: layer 2 forward, hn[3]: layer 2 backward

# For classification: last layer, both directions
last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)   # [batch, 512]
```

### When to Use Bidirectional

**Use BiLSTM/BiGRU when:**
- The full sequence is available at inference time (not streaming)
- Context from both directions would help understanding
- Text classification, NER, part-of-speech tagging, semantic role labeling

**Cannot use bidirectional when:**
- Generating output one token at a time (language modeling, translation)
- Real-time/streaming processing where future is unknown

BERT is essentially a deeply stacked bidirectional transformer — it replaced BiLSTM as the standard encoder for NLP tasks.

---

## Encoder-Decoder (Seq2Seq)

### The Problem

Standard RNNs output one value per input timestep — input length = output length. Translation, summarization, and speech recognition all need **different-length** inputs and outputs.

### The Solution

Split the task into two phases:

**Encoder:** Read the entire input sequence, compress everything into a fixed-size **context vector** (the final hidden state).

**Decoder:** Generate the output sequence one token at a time, conditioned on the context vector.

```
INPUT:  "The dog barked"  (3 tokens)
OUTPUT: "Le chien a aboyé"  (4 tokens — different length)

ENCODER (LSTM):
  "The"   → [LSTM] → h₁
  "dog"   → [LSTM] → h₂
  "barked"→ [LSTM] → h₃ = c (context vector, 256-dim)

DECODER (LSTM, different weights):
  c + <START>    → [LSTM] → "Le"
  c + "Le"       → [LSTM] → "chien"
  c + "chien"    → [LSTM] → "a"
  c + "a"        → [LSTM] → "aboyé"
  c + "aboyé"    → [LSTM] → <END>
```

### Implementation

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, batch_first=True)
    
    def forward(self, src):
        embedded = self.embed(src)
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell   # Context: final hidden + cell state

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.output_fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, token, hidden, cell):
        embedded = self.embed(token.unsqueeze(1))           # [batch, 1, embed]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.output_fc(output.squeeze(1))      # [batch, vocab_size]
        return prediction, hidden, cell
```

### Teacher Forcing

During training, instead of feeding the decoder's own (possibly wrong) predictions back, feed the **ground truth** previous token. This stabilizes training:

```python
# With teacher forcing
for t in range(target_len):
    pred, hidden, cell = decoder(target[t], hidden, cell)   # Feed ground truth
    loss += criterion(pred, target[t+1])

# Without teacher forcing (inference mode)
token = start_token
for t in range(max_len):
    pred, hidden, cell = decoder(token, hidden, cell)
    token = pred.argmax()   # Feed own prediction
```

A common strategy: use teacher forcing with probability p during training, gradually decrease p as training progresses (scheduled sampling).

### The Bottleneck Problem

The context vector `c` must compress the **entire input sequence** into a single fixed-size vector (e.g., 512 dimensions). For short sequences, fine. For long sequences:

```
"The quick brown fox jumped over the lazy dog near the old stone bridge"
→ compressed into 512 numbers
→ decoder must regenerate all 15 words from this alone
→ information loss is inevitable, especially for early words
```

This is why translations of long sentences were poor before attention. The decoder has to "remember" everything about the input from one vector — early words are often forgotten.

---

## Attention Mechanism (2015)

### The Fix

Bahdanau et al. (2015) — "Neural Machine Translation by Jointly Learning to Align and Translate."

Instead of compressing the input into one vector, **keep all encoder hidden states** and let the decoder **dynamically choose which parts of the input to focus on** at each decoding step.

```
WITHOUT ATTENTION:
  Encoder → c (one vector) → Decoder uses c for every output token

WITH ATTENTION:
  Encoder → [h₁, h₂, ..., hₙ] (all hidden states, kept)
  At each decoder step t:
    - Compute attention score: eₜᵢ = score(sₜ, hᵢ)  for each encoder state hᵢ
    - Normalize: αₜᵢ = softmax(eₜᵢ)               (attention weights, sum to 1)
    - Compute context: cₜ = Σᵢ αₜᵢ · hᵢ            (weighted sum of encoder states)
    - Decode: sₜ = LSTM(sₜ₋₁, yₜ₋₁, cₜ)           (different context at each step)
```

The attention weights `αₜᵢ` are learned — the model learns which input positions to look at when generating each output token.

### Visualizing Attention

Attention weights are interpretable. For English→French translation:
```
                  "Le"  "chien"  "a"   "aboyé"   <END>
"The"           [ 0.9    0.0    0.05    0.0       0.05 ]
"dog"           [ 0.0    0.95   0.0     0.0       0.05 ]
"barked"        [ 0.0    0.0    0.1     0.85      0.05 ]
```

When generating "Le", the decoder attends to "The". When generating "chien", it attends to "dog". When generating "aboyé", it attends to "barked". The model learned word alignment from data.

### Attention Score Functions

Three common ways to compute the relevance of encoder state `hᵢ` to decoder state `sₜ`:

```python
# Dot product (fast, Luong et al.):
score = torch.dot(sₜ, hᵢ)

# Scaled dot product (used in Transformer):
score = torch.dot(sₜ, hᵢ) / sqrt(hidden_dim)

# Additive / Concat (original Bahdanau, more parameters):
score = v · tanh(W₁·sₜ + W₂·hᵢ)
```

### Why This Matters for Transformers

Attention in RNN context:
- Encoder-decoder attention: decoder attends to encoder states
- Context is a weighted sum of encoder states
- Weights determined by query (decoder state) vs keys (encoder states)

The Transformer took this idea and:
1. Made attention the **primary** (not supplementary) operation
2. Applied it to the **same sequence** (self-attention, not just encoder-decoder)
3. Ran multiple attention heads in parallel
4. **Removed the RNN entirely** — no recurrence needed

The full Transformer mechanism and self-attention are covered in `transformer.md`. Attention here is introduced as the bridge — it was born in the RNN world, then transcended it.