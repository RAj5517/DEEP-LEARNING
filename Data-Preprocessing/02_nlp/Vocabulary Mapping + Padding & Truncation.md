# Vocabulary Mapping + Padding & Truncation

After tokenization, every token must be converted to a number the model can process, and all sequences must be brought to a uniform length.

---

# Part 1 — Vocabulary Mapping

## 1. What is a Vocabulary?

A vocabulary is a fixed dictionary mapping every known token to a unique integer index.

```
"hello"  → 42
"world"  → 87
"[PAD]"  → 0
"[UNK]"  → 1
```

---

## 2. Building a Vocabulary from Scratch

```python
from collections import Counter

corpus = ["hello world", "hello there", "world is great"]
tokens = " ".join(corpus).split()

# Count frequencies
freq = Counter(tokens)

# Build vocab with special tokens first
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
vocab = {tok: i for i, tok in enumerate(special_tokens)}

# Add tokens that appear at least min_freq times
min_freq = 1
for token, count in freq.most_common():
    if count >= min_freq and token not in vocab:
        vocab[token] = len(vocab)

# Reverse mapping: index → token
idx2token = {i: t for t, i in vocab.items()}

print(vocab)
# {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'hello': 4, 'world': 5, ...}
```

---

## 3. Token → Index (Encoding)

```python
UNK_IDX = vocab["[UNK]"]

def encode(text, vocab):
    return [vocab.get(token, UNK_IDX) for token in text.split()]

indices = encode("hello unknown_word world", vocab)
# → [4, 1, 5]   (1 = [UNK] for unknown_word)
```

---

## 4. Index → Token (Decoding)

```python
def decode(indices, idx2token):
    return " ".join(idx2token.get(i, "[UNK]") for i in indices)

text = decode([4, 1, 5], idx2token)
# → "hello [UNK] world"
```

---

## 5. Using HuggingFace Tokenizers (Production)

Modern pretrained models come with their vocabulary baked in — no need to build manually.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encode: text → token ids
encoded = tokenizer.encode("Hello world", add_special_tokens=True)
# → [101, 7592, 2088, 102]

# Decode: token ids → text
decoded = tokenizer.decode(encoded)
# → "[CLS] hello world [SEP]"

# Batch encoding
batch = tokenizer(
    ["Hello world", "How are you?"],
    padding=True,
    truncation=True,
    max_length=16,
    return_tensors="pt"
)
# Returns: input_ids, attention_mask, token_type_ids
```

---

## 6. Handling Out-of-Vocabulary (OOV) Tokens

| Strategy | How | When |
|---|---|---|
| `[UNK]` token | Map all unknown tokens to index 1 | Word-level tokenizers |
| Subword fallback | Split unknown word into subwords | BPE / WordPiece / SentencePiece |
| Character fallback | Decompose to characters | Character-level models |
| Ignore | Skip unknown tokens | Some classical NLP |

---

# Part 2 — Padding & Truncation

## 7. Why Padding is Needed

Neural networks process data in batches. All sequences in a batch must be the **same length**. Since sentences vary in length, shorter ones are padded and longer ones are truncated.

```
Sequence 1:  [4, 7, 2]              → length 3
Sequence 2:  [4, 7, 2, 9, 11, 6]   → length 6
Sequence 3:  [4, 7]                 → length 2

After padding to max_len=6:
Sequence 1:  [4, 7, 2,  0, 0, 0]
Sequence 2:  [4, 7, 2,  9, 11, 6]
Sequence 3:  [4, 7, 0,  0, 0, 0]
```

---

## 8. Padding Strategies

```python
import torch
from torch.nn.utils.rnn import pad_sequence

sequences = [
    torch.tensor([4, 7, 2]),
    torch.tensor([4, 7, 2, 9, 11, 6]),
    torch.tensor([4, 7]),
]

# Post-padding (pad at end) — default, recommended for most models
padded = pad_sequence(sequences, batch_first=True, padding_value=0)

# Pre-padding (pad at start) — sometimes used for decoder-only models
def pre_pad(seqs, pad_val=0):
    max_len = max(len(s) for s in seqs)
    return torch.stack([
        torch.cat([torch.full((max_len - len(s),), pad_val), s])
        for s in seqs
    ])
```

---

## 9. Truncation Strategies

```python
max_length = 512  # BERT's maximum sequence length

# Truncate from the end (most common)
truncated = token_ids[:max_length]

# Truncate from the start (useful if answer is at the end)
truncated = token_ids[-max_length:]

# HuggingFace handles both
tokenizer(text, truncation=True, max_length=512, truncation_side="right")
```

**Caution:** Truncation discards information. For tasks where the end of a document matters (e.g., summarization), truncating from the right loses the conclusion.

---

## 10. The Attention Mask

When padding is added, the model must know which tokens are real and which are padding — it should not attend to padding positions.

```python
# input_ids:      [101, 7592, 2088, 102,   0,   0]
# attention_mask: [  1,    1,    1,   1,   0,   0]
# 1 = real token, 0 = padding — model ignores these positions
```

HuggingFace generates attention masks automatically:

```python
encoded = tokenizer(
    ["Hello world", "Hi"],
    padding=True,
    return_tensors="pt"
)
print(encoded["input_ids"])
#  tensor([[ 101, 7592, 2088,  102],
#          [ 101, 7632,  102,    0]])

print(encoded["attention_mask"])
#  tensor([[1, 1, 1, 1],
#          [1, 1, 1, 0]])
```

---

## 11. Choosing max_length

| Model | Max Token Limit |
|---|---|
| BERT (base) | 512 |
| GPT-2 | 1024 |
| GPT-3 / GPT-4 | 4096–128K |
| LLaMA 2 | 4096 |
| LLaMA 3 | 8192 |
| T5 | 512 (input), 512 (output) |

For custom LSTMs: use the 95th–99th percentile of sequence lengths in your dataset — not the maximum, which may be an outlier.

```python
import numpy as np
lengths = [len(tokenizer.encode(text)) for text in corpus]
max_length = int(np.percentile(lengths, 95))
print(f"Recommended max_length: {max_length}")
```