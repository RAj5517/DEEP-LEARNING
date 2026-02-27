# Encoder-Decoder Transformers (Seq2Seq)

> "Understand the input completely. Then generate the output from scratch."

---

## What It Does

Encoder-Decoder (also called Seq2Seq) models do two things:
1. **Encoder** reads and fully understands the input (bidirectional attention)
2. **Decoder** generates the output token by token, while constantly referring back to the encoder's understanding via **Cross-Attention**

This makes them perfect for tasks where input and output are two **different** sequences — like translation (English → French) or summarization (long article → short summary).

---

## Architecture

```
INPUT SEQUENCE
│
├── Token Embedding + Positional Encoding
│
└── ENCODER (× N layers)
      ├── Multi-Head Self-Attention (Bidirectional)
      ├── Add & LayerNorm
      ├── Feed Forward Network
      └── Add & LayerNorm
│
└── Encoder Output → rich contextual vectors for every input token
         │
         │  ← passed as K and V into decoder's Cross-Attention
         ↓
OUTPUT SEQUENCE (generated so far)
│
├── Token Embedding + Positional Encoding
│
└── DECODER (× N layers)
      ├── Masked Self-Attention       (only sees past output tokens)
      ├── Add & LayerNorm
      ├── Cross-Attention             Q from decoder · K,V from encoder
      │     └── "Which input tokens matter for generating this output token?"
      ├── Add & LayerNorm
      ├── Feed Forward Network
      └── Add & LayerNorm
│
└── Linear + Softmax → Next output token
```

---

## Three Types of Attention at Play

```
Attention in Encoder-Decoder
│
├── Encoder Self-Attention       input tokens attend to each other
│     └── Bidirectional — full context of input
│
├── Decoder Masked Self-Attention  output tokens attend to past output
│     └── Masked — can't peek at future output tokens
│
└── Cross-Attention              BRIDGE between encoder and decoder
      ├── Query  →  from decoder (what output am I generating?)
      ├── Key    →  from encoder (what did the input say?)
      └── Value  →  from encoder (what information to pull from input?)
```

Cross-Attention is what makes this architecture unique.
Without it, the decoder has no idea what the input was.

---

## Use Cases

```
Encoder-Decoder Use Cases
│
├── Machine Translation          English → French · Hindi → Spanish
├── Text Summarization           1000 word article → 3 sentence summary
├── Question Answering           given context + question → generate answer
├── Text Rewriting               paraphrase · grammar correction · simplification
├── Code Translation             Python → JavaScript
└── Dialogue / Chatbot           given conversation history → generate reply
```

---

## Main Models

### T5 — Text-to-Text Transfer Transformer (2019) — Google
The "unify everything" model. Every NLP task is framed as text-in, text-out:
- Classification → `"classify: I love this movie"` → `"positive"`
- Translation → `"translate English to French: Hello"` → `"Bonjour"`
- Summarization → `"summarize: [article]"` → `"[summary]"`

Same model, same loss function, same format for all tasks. Trained on the Colossal Clean Crawled Corpus (C4) — 750GB of clean web text.

- Sizes: T5-Small (60M) → T5-11B (11B parameters)
- Fine-tuned versions outperform task-specific models on most benchmarks
- Paper: https://arxiv.org/abs/1910.10683

---

### BART (2019) — Facebook AI
**Bidirectional and Auto-Regressive Transformer**

Pre-trained by corrupting text in various ways and learning to reconstruct the original:
- Token masking (like BERT)
- Sentence permutation
- Document rotation
- Token deletion

Particularly strong at **summarization** — BART fine-tuned on CNN/DailyMail became the go-to summarization model for years.

- Paper: https://arxiv.org/abs/1910.13461

---

### mT5 — Multilingual T5 (2020) — Google
T5 trained on 101 languages. Same architecture, same text-to-text format — but multilingual. Useful for cross-lingual tasks: train on English, run on Hindi.

Paper: https://arxiv.org/abs/2010.11934

---

### Pegasus (2019) — Google
Designed specifically for summarization. Pre-training objective: Gap Sentence Generation — removes important sentences from documents and makes the model predict them. Essentially pre-trains on a task that resembles summarization directly.

Paper: https://arxiv.org/abs/1912.08777

---

### MarianMT — Helsinki NLP
A collection of 1000+ pre-trained translation models covering hundreds of language pairs. Built on the Marian C++ framework. Fast, lightweight, and available directly on HuggingFace for almost any language pair you need.

---

## T5 vs BART — Quick Comparison

```
T5                              BART
│                               │
├── Encoder-Decoder             ├── Encoder-Decoder
├── Pre-train: denoising        ├── Pre-train: text corruption + reconstruct
├── Format: text-to-text        ├── Format: standard seq2seq
├── Great at: all NLP tasks     ├── Great at: summarization · generation
└── Fine-tune: easy + unified   └── Fine-tune: task specific
```

---

## Why Not Just Use a Decoder-Only Model?

Large decoder-only models (GPT-4, Claude) can also summarize and translate — so why use encoder-decoder?

- Encoder-decoder is **more efficient** for fixed input → output tasks
- The encoder processes input **once**, decoder attends to it efficiently via cross-attention
- For smaller, specialized deployments, a fine-tuned T5 or BART is cheaper and faster than running a 70B decoder model
- Encoder-decoder models still dominate **production translation** pipelines

---

## Key Papers

| Paper | Link |
|-------|------|
| Original Transformer (2017) | https://arxiv.org/abs/1706.03762 |
| T5 (2019) | https://arxiv.org/abs/1910.10683 |
| BART (2019) | https://arxiv.org/abs/1910.13461 |
| Pegasus (2019) | https://arxiv.org/abs/1912.08777 |
| mT5 (2020) | https://arxiv.org/abs/2010.11934 |