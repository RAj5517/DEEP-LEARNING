# 🧠 NLP — Text Preprocessing

> Neural networks do not understand language.
> They understand numbers.
> Everything between raw text and model input is **NLP preprocessing**.

---

## 🔁 Pipeline at a Glance

```
Raw Text
   ↓
Text Cleaning        (optional — task dependent)
   ↓
Tokenization         (split text into units)
   ↓
Vocabulary Mapping   (token → integer index)
   ↓
Padding / Truncation (uniform sequence length)
   ↓
Embedding Layer      (integer → dense vector)
   ↓
Model  (LSTM / Transformer / LLM)
```

---

## 📂 Structure

| File | Covers |
|---|---|
| `01_text_cleaning.md` | Lowercasing, stopwords, stemming, when to skip |
| `02_tokenization.md` | Character, Word, BPE, WordPiece, SentencePiece |
| `03_vocab_mapping_padding.md` | token→index, OOV, padding, truncation, attention mask |
| `04_embedding_layer.md` | Learned, GloVe, BERT, positional encoding |

---

## ❓ Why This Matters

| Problem | Caused By |
|---|---|
| Model sees garbage | No cleaning — HTML, URLs, noise in input |
| OOV tokens everywhere | Wrong tokenizer or no subword splitting |
| Training crashes | No padding — variable-length sequences can't batch |
| Slow convergence | Random embeddings — no pretrained initialization |
| Model ignores order | No positional encoding in Transformers |

---

## ⚡ When to Apply Each Step

| Step | LSTM | BERT / GPT | Classical NLP |
|---|---|---|---|
| Text Cleaning | Light | Skip / minimal | Aggressive |
| Tokenization | Word or Subword | WordPiece / BPE | Word-level |
| Vocab Mapping | Custom vocab | Pretrained tokenizer | Custom vocab |
| Padding | Yes | Yes | Not needed |
| Embeddings | GloVe or learned | Built-in contextual | TF-IDF / BoW |

---

## 🔬 Core Idea

Raw text → tokens → integers → vectors → model.

Every step serves one purpose:
**convert human language into a form that gradient descent can learn from.**

---

*For deep breakdowns, math, and code — refer to the individual files above.*

┌──────────────────────────────────────────┐
  │               RAW TEXT                   │
  └──────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │       1. TEXT CLEANING  (optional)       │
  └──────────────────────────────────────────┘
  │
  ├── Lowercasing
  ├── Remove Punctuation
  ├── Remove HTML / URLs
  ├── Remove Stopwords          (traditional NLP only)
  ├── Spelling Correction
  ├── Expand Contractions
  └── Stemming / Lemmatization  (traditional NLP only)
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │          2. TOKENIZATION                 │
  └──────────────────────────────────────────┘
  │
  ├── Character-level
  │     splits into individual characters
  │
  ├── Word-level
  │     splits on whitespace & punctuation
  │
  └── Subword  ← modern standard
        ├── BPE  (Byte-Pair Encoding)   → GPT family
        ├── WordPiece                   → BERT family
        └── SentencePiece              → T5 · LLaMA
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │        3. VOCABULARY MAPPING             │
  └──────────────────────────────────────────┘
  │
  ├── Build vocab   token → index
  │     char2idx = {tok: i for i, tok in enumerate(vocab)}
  │
  ├── Encode        token → integer id
  ├── Decode        integer id → token
  │
  └── Special Tokens
        ├── [PAD]  → padding
        ├── [UNK]  → unknown / OOV
        ├── [CLS]  → sentence start   (BERT)
        └── [SEP]  → sentence end     (BERT)
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │      4. PADDING / TRUNCATION             │
  └──────────────────────────────────────────┘
  │
  ├── Padding       (shorter sequences)
  │     ├── Post-pad  →  [4, 7, 2, 0, 0]   ← default
  │     └── Pre-pad   →  [0, 0, 4, 7, 2]
  │
  └── Truncation    (longer sequences)
        ├── From end    →  seq[:max_len]
        └── From start  →  seq[-max_len:]
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │          5. EMBEDDING LAYER              │
  └──────────────────────────────────────────┘
  │
  ├── Learned from scratch
  │     nn.Embedding(vocab_size, embedding_dim)
  │
  ├── Pretrained Static
  │     ├── Word2Vec
  │     ├── GloVe
  │     └── FastText
  │
  ├── Pretrained Contextual
  │     ├── BERT  (768-dim)
  │     └── GPT   (768–12288-dim)
  │
  └── Positional Encoding       (Transformers only)
        ├── Sinusoidal           → original Transformer
        └── Learned              → BERT · GPT
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │       MODEL (LSTM / Transformer / LLM)   │
  └──────────────────────────────────────────┘