# Tokenization

Tokenization splits raw text into discrete units called **tokens** — the atomic input units your model sees. The choice of tokenizer directly shapes vocabulary size, model capacity, and how well the model handles rare or unknown words.

---

## 1. Character-level Tokenization

Splits text into individual characters.

```python
text = "hello"
tokens = list(text)
# → ['h', 'e', 'l', 'l', 'o']
```

**Vocabulary size:** ~100–300 (all printable characters)

**Pros:**
- Zero out-of-vocabulary (OOV) problem — every character exists
- Handles any language, typos, code

**Cons:**
- Very long sequences — models must learn to compose meaning from characters
- Slow training, harder to learn semantics

**Used in:** Character-level RNNs, some multilingual models, DNA/protein sequence models

---

## 2. Word-level Tokenization

Splits text on whitespace and punctuation boundaries.

```python
import nltk
tokens = nltk.word_tokenize("Hello, world! How are you?")
# → ['Hello', ',', 'world', '!', 'How', 'are', 'you', '?']
```

**Vocabulary size:** Tens of thousands to millions

**Pros:**
- Intuitive — tokens map directly to words
- Short sequences

**Cons:**
- Massive vocabulary → huge embedding tables
- OOV problem: `"unhappiness"` not in vocab if only `"happy"` was seen
- Fails on morphologically rich languages (Finnish, Turkish)

**Used in:** Older NLP models, Word2Vec, GloVe embeddings

---

## 3. Subword Tokenization

The dominant approach in modern NLP. Splits words into meaningful sub-units — common words stay whole, rare words are split into pieces.

```
"unhappiness" → ["un", "happiness"]
"tokenization" → ["token", "ization"]
"ChatGPT" → ["Chat", "G", "PT"]
```

**Vocabulary size:** 30,000–50,000 (a sweet spot)

**Pros:**
- Handles OOV words — any word can be represented as subword pieces
- Compact vocabulary vs word-level
- Captures morphology (prefixes, suffixes)

**Cons:**
- Tokens don't always align with human intuition
- One word → multiple tokens increases sequence length

### 3.1 Byte-Pair Encoding (BPE)

Used by: **GPT-2, GPT-3, GPT-4, RoBERTa, LLaMA**

Starts with characters, iteratively merges the most frequent adjacent pairs.

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["corpus.txt"], vocab_size=50000, min_frequency=2)
tokenizer.save_model("bpe_model")

# Using HuggingFace
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("unhappiness")
# → ['un', 'happiness']
```

### 3.2 WordPiece

Used by: **BERT, DistilBERT, ELECTRA**

Similar to BPE but merges pairs that maximize likelihood of the training data. Prefixes subword pieces with `##`.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unhappiness")
# → ['un', '##happiness']

# Full encoding with special tokens
encoded = tokenizer("Hello world", return_tensors="pt")
# adds [CLS] at start, [SEP] at end
```

### 3.3 SentencePiece

Used by: **T5, ALBERT, mT5, LLaMA**

Language-agnostic — treats the raw byte stream directly (no whitespace pre-tokenization required). Works well for Chinese, Japanese, Korean.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("sentencepiece.model")
tokens = sp.encode("Hello world", out_type=str)
# → ['▁Hello', '▁world']  (▁ marks word start)
```

---

## 4. Special Tokens

Most modern tokenizers add special control tokens:

| Token | Purpose | Model |
|---|---|---|
| `[CLS]` | Classification token (sentence start) | BERT |
| `[SEP]` | Separator between sentences | BERT |
| `[PAD]` | Padding to fixed length | BERT, GPT |
| `[UNK]` | Unknown / OOV token | BERT |
| `[MASK]` | Masked token for MLM training | BERT |
| `<s>` / `</s>` | Sequence start / end | RoBERTa, T5 |
| `<pad>` | Padding | T5, LLaMA |

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer("Hello world")
# {'input_ids': [101, 7592, 2088, 102]}
# 101 = [CLS], 102 = [SEP]
```

---

## 5. Comparison Table

| Method | Vocab Size | OOV Handling | Sequence Length | Used In |
|---|---|---|---|---|
| Character-level | ~100–300 | None | Very long | Char RNNs |
| Word-level | 50K–1M+ | Poor | Short | Word2Vec, GloVe |
| BPE | 30K–50K | Excellent | Medium | GPT family |
| WordPiece | 30K | Excellent | Medium | BERT family |
| SentencePiece | 32K | Excellent | Medium | T5, LLaMA |