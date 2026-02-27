# Transformer Architecture — Master Overview

The Transformer was introduced in 2017 by Google Brain in the paper **"Attention Is All You Need"**.
It replaced RNNs and LSTMs entirely for most tasks by removing sequential processing and replacing it with **Attention** — every token looks at every other token simultaneously.

---

## Core Idea

Before Transformers, models read sequences one word at a time (RNN, LSTM).
Transformers read the **entire sequence at once** and let every word decide which other words matter most to understand itself. That mechanism is called **Self-Attention**.

---

## Common Building Blocks (All Transformers share these)

```
Core Components
│
├── Embedding
│     ├── Token Embedding        word → vector of numbers
│     └── Positional Encoding    inject position info (sin/cos waves)
│
├── Attention Mechanism
│     ├── Self-Attention         Q · K · V — every token attends to others
│     ├── Multi-Head Attention   run N attention heads in parallel
│     ├── Masked Attention       block future tokens (used in decoders)
│     └── Cross-Attention        decoder queries encoder output (seq2seq only)
│
├── Feed Forward Network         2 linear layers + activation (per token)
│
├── Normalization
│     ├── LayerNorm              original BERT · GPT-2
│     └── RMSNorm                LLaMA · Mistral · modern LLMs
│
├── Residual Connections         add input back to output of each sub-layer
│
└── Output Head
      ├── Softmax + Linear       for generation (next token prediction)
      └── Classifier Head        for classification ([CLS] token → label)
```

---

## Types of Transformers

```
Transformer Types
│
├── Encoder-Only                 → encoder-only.md
│     ├── Attention              Bidirectional — sees full input at once
│     ├── Use                    Classification · NER · Embeddings · Search
│     └── Models                 BERT · RoBERTa · DeBERTa
│
├── Decoder-Only                 → decoder-only.md
│     ├── Attention              Masked — left to right, no peeking
│     ├── Use                    Generation · Chat · Code · Completion
│     └── Models                 GPT-4 · Claude · LLaMA · Mistral
│
├── Encoder-Decoder (Seq2Seq)    → encoder-decoder.md
│     ├── Attention              Bidirectional + Masked + Cross-Attention
│     ├── Use                    Translation · Summarization
│     └── Models                 T5 · BART
│
├── Vision Transformer           → vision-transformer.md
│     ├── Attention              Patches as tokens · Self-Attention over image
│     ├── Use                    Image Classification · Detection · Segmentation
│     └── Models                 ViT · Swin · DeiT
│
├── Multimodal Transformer       → multimodal.md
│     ├── Attention              Cross-modal attention (text + image + audio)
│     ├── Use                    Visual QA · Captioning · Text-to-Image
│     └── Models                 CLIP · GPT-4V · Gemini · LLaVA
│
└── Domain-Specific
      ├── Code                   → domain-code.md
      ├── Biology / Protein      → domain-biology.md
      ├── Chemistry              → domain-chemistry.md
      ├── Audio / Speech         → domain-audio.md
      └── Time Series            → domain-timeseries.md
```

---

## Attention Formula

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

Q  →  Query   — what am I looking for?
K  →  Key     — what do I contain?
V  →  Value   — what do I give if selected?
√d →  scaling factor to prevent vanishing gradients
```

---

## Key Papers

| Paper | Year | Link |
|-------|------|------|
| Attention Is All You Need | 2017 | https://arxiv.org/abs/1706.03762 |
| BERT | 2018 | https://arxiv.org/abs/1810.04805 |
| GPT-3 | 2020 | https://arxiv.org/abs/2005.14165 |
| ViT | 2020 | https://arxiv.org/abs/2010.11929 |
| T5 | 2019 | https://arxiv.org/abs/1910.10683 |
| CLIP | 2021 | https://arxiv.org/abs/2103.00020 |