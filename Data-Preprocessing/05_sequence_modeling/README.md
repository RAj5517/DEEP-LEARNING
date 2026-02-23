# 🔄 Sequence Modeling — Preprocessing Pipeline

> The model doesn't read text or music.
> It predicts the next token in a sequence.
> Everything between raw data and that prediction is **sequence preprocessing**.

---

## 🔁 Pipeline at a Glance

```
Raw Text / Music Data
   ↓
Tokenization          (split into discrete units)
   ↓
Vocabulary Mapping    (token → integer index)
   ↓
Sequence Windowing    (sliding windows over encoded data)
   ↓
(Input, Target) Pairs (target = input shifted right by 1)
   ↓
Shuffle               (break ordering bias)
   ↓
Batch                 (group for parallel GPU computation)
   ↓
LSTM / Transformer
```

---

## 📂 Structure

| File | Covers |
|---|---|
| `01_tokenization_vocab.md` | Char/word/subword/MIDI tokens, vocab building, special tokens |
| `02_sequence_windowing.md` | Sliding window, input/target pairs, TBPTT, causal mask |
| `03_shuffle_batch.md` | DataLoader, stateful batching, gradient clipping, metrics |

---

## ❓ Why This Matters

| Problem | Caused By |
|---|---|
| Model predicts garbage | No vocabulary mapping — raw strings into model |
| Learns ordering, not patterns | No shuffle — sees same order every epoch |
| Exploding gradients | No gradient clipping — LSTMs are especially vulnerable |
| Model peeks at the future | No causal mask in Transformer |
| Inconsistent batch shapes | No drop_last — final batch is smaller |

---

## ⚡ When to Apply Each Step

| Step | LSTM | Transformer | Music Gen |
|---|---|---|---|
| Char tokenization | ✅ | ⚠️ Subword preferred | ✅ |
| Vocabulary mapping | ✅ | ✅ | ✅ |
| Sliding window | ✅ | ✅ | ✅ |
| TBPTT | ✅ | ❌ | ✅ |
| Causal mask | ❌ | ✅ | ✅ |
| Shuffle | ✅ | ✅ | ✅ |
| Stateful batching | ✅ | ❌ | ✅ |
| Gradient clipping | ✅ | ⚠️ | ✅ |

---

## 🔬 Core Idea

Sequence modeling has one objective:

**Given everything seen so far → predict the next token.**

The entire preprocessing pipeline exists to create that prediction task from raw data — and to deliver it to the model efficiently.

---

*For deep breakdowns, math, and code — refer to the individual files above.*

┌──────────────────────────────────────────┐
  │       RAW TEXT / MUSIC DATA              │
  └──────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │          1. TOKENIZATION                 │
  └──────────────────────────────────────────┘
  │
  ├── Character-level   →  ["h","e","l","l","o"]    ← LSTM standard
  ├── Word-level        →  ["hello","world"]
  ├── Subword (BPE)     →  ["hel","lo"]             ← Transformer standard
  └── Music (MIDI)      →  [NOTE_ON, TIME_SHIFT, NOTE_OFF ...]
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │        2. VOCABULARY MAPPING             │
  └──────────────────────────────────────────┘
  │
  ├── Build vocab       →  all unique tokens
  ├── char2idx          →  {"h":0, "e":1, "l":2 ...}
  ├── idx2char          →  {0:"h", 1:"e", 2:"l" ...}
  │
  └── Special tokens
        ├── <PAD>   →  padding
        ├── <UNK>   →  unknown / OOV
        ├── <BOS>   →  beginning of sequence
        └── <EOS>   →  end of sequence
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │       3. SEQUENCE WINDOWING              │
  └──────────────────────────────────────────┘
  │
  ├── Sliding window  (overlapping)
  │     stride = 1  →  maximum samples
  │
  ├── Non-overlapping
  │     stride = seq_len  →  efficient, less redundancy
  │
  ├── seq_len guide
  │     ├── Char LSTM      →  64–256
  │     ├── Word LSTM      →  35–100
  │     └── Transformer    →  1024–8192
  │
  └── Causal Mask  (Transformers)
        upper-triangular -inf mask → no future peeking
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │    4. CREATE (INPUT, TARGET) PAIRS       │
  └──────────────────────────────────────────┘
  │
  │   Input  →  "hell"   →  [h, e, l, l]
  │   Target →  "ello"   →  [e, l, l, o]
  │
  │   Target = Input shifted right by 1 position
  │   Model learns:  given token[i], predict token[i+1]
  │
  └── TBPTT  (LSTMs on very long sequences)
        detach hidden state every N steps
        stop gradient from flowing through entire history
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │            5. SHUFFLE                   │
  └──────────────────────────────────────────┘
  │
  ├── Shuffle every epoch       →  prevents order memorization
  ├── DataLoader(shuffle=True)  →  automatic
  └── Skip shuffle              →  stateful LSTM / time series batching
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │             6. BATCH                    │
  └──────────────────────────────────────────┘
  │
  ├── batch_size    →  64–256  (standard)
  ├── drop_last     →  True    (uniform batch size)
  ├── pin_memory    →  True    (faster GPU transfer)
  ├── num_workers   →  4       (parallel loading)
  │
  ├── Gradient Clipping  →  clip_grad_norm_(max_norm=1.0)
  └── Gradient Accumulation  →  simulate larger batch on small GPU
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │       LSTM / TRANSFORMER / LLM           │
  └──────────────────────────────────────────┘
  │
  └── Metrics
        ├── Cross-Entropy Loss   →  -log P(next token)
        ├── Perplexity           →  exp(loss)
        └── Bits per character   →  loss / log(2)