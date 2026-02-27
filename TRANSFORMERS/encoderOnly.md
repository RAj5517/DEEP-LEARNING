# Encoder-Only Transformers

> "Read the whole sentence. Understand everything. Don't generate anything."

---

## What It Does

An Encoder-Only model takes a full input sequence and produces a **contextual representation** of every token. It sees the entire sentence — left and right — before deciding what each word means.

The word "bank" in "I sat by the river bank" means something completely different from "I went to the bank to withdraw money." An encoder sees both sides of the word before encoding it — so it gets it right.

This is called **Bidirectional Attention**.

---

## Architecture

```
Input
│
├── Token Embedding              word → dense vector
├── Positional Encoding          inject position (token 1, 2, 3...)
│
└── × N Encoder Layers
      ├── Multi-Head Self-Attention
      │     └── Bidirectional — every token attends to ALL other tokens
      ├── Add & LayerNorm
      ├── Feed Forward Network
      └── Add & LayerNorm
│
└── Output
      ├── [CLS] token vector     → Classification Head → label
      ├── All token vectors      → NER · POS tagging · token-level tasks
      └── Pooled output          → Sentence embeddings · semantic search
```

---

## Attention Type

**Bidirectional Self-Attention** — no masking.
Every token can attend to every other token in both directions.

```
"The cat sat on the mat"
         ↑
    "sat" attends to "cat" (left) AND "mat" (right) simultaneously
```

This is why encoder-only models understand context so deeply — but they can't generate text because they'd need future tokens that don't exist yet during generation.

---

## Pre-Training Objectives

Encoder models learn through:

**Masked Language Modeling (MLM)**
- Randomly mask 15% of tokens
- Model predicts the masked word using both sides
- `"The [MASK] sat on the mat"` → predict "cat"

**Next Sentence Prediction (NSP)** (BERT only)
- Given two sentences, predict if B follows A
- Teaches document-level understanding

---

## Use Cases

```
Encoder-Only Use Cases
│
├── Text Classification          sentiment · spam · topic · intent
├── Named Entity Recognition     find Person · Location · Organization in text
├── Question Answering           find answer span in a given paragraph
├── Semantic Search              encode query + docs → compare similarity
├── Sentence Embeddings          compress sentence into a single vector
└── Natural Language Inference   does sentence A entail / contradict sentence B?
```

---

## Main Models

### BERT (2018) — Google
**Bidirectional Encoder Representations from Transformers**

The model that changed NLP forever. Pre-trained on Wikipedia + BooksCorpus using MLM + NSP. Fine-tune on any downstream task with minimal extra layers.

- 12 layers · 110M parameters (Base) / 24 layers · 340M (Large)
- First model to achieve state-of-the-art on 11 NLP benchmarks simultaneously
- Paper: https://arxiv.org/abs/1810.04805

```
BERT Architecture
Input: [CLS] The cat sat [MASK] the mat [SEP]
         ↓
  12 Transformer Encoder Layers (bidirectional attention)
         ↓
  [CLS] vector → fine-tune for classification
  Token vectors → fine-tune for NER / QA
```

---

### RoBERTa (2019) — Facebook AI
**Robustly Optimized BERT Pretraining Approach**

Same architecture as BERT. But trained better:
- Removed NSP (found it harmful)
- Trained 10× longer on 10× more data
- Larger batch sizes
- Dynamic masking (mask changes each epoch)

Result: Significantly outperforms BERT on every benchmark.
Paper: https://arxiv.org/abs/1907.11692

---

### DeBERTa (2020) — Microsoft
**Decoupled Attention with Enhanced Mask Decoder**

Key innovation: separates content and position into two different attention matrices instead of combining them. Model attends to content AND position independently, then combines.

- Outperforms RoBERTa and even larger models on many benchmarks
- Current go-to for NLP classification tasks
- Paper: https://arxiv.org/abs/2006.03654

---

### Other Notable Models

| Model | By | Key Difference |
|-------|----|----------------|
| ALBERT | Google | Parameter sharing across layers — smaller size |
| ELECTRA | Google | Replaced MLM with token discrimination — more efficient training |
| DistilBERT | HuggingFace | 40% smaller BERT · 60% faster · 97% performance |
| XLM-RoBERTa | Facebook | Multilingual — 100 languages |

---

## When to Use Encoder-Only

Use encoder-only when:
- You need to **understand** text, not generate it
- Task is classification, extraction, or similarity
- You want **embeddings** for search or clustering

Do NOT use when:
- You need to generate text (use decoder-only)
- You need translation or summarization (use encoder-decoder)

---

## Key Paper

| Paper | Link |
|-------|------|
| BERT (2018) | https://arxiv.org/abs/1810.04805 |
| RoBERTa (2019) | https://arxiv.org/abs/1907.11692 |
| DeBERTa (2020) | https://arxiv.org/abs/2006.03654 |
| ELECTRA (2020) | https://arxiv.org/abs/2003.10555 |