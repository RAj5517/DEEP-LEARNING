# Tokenization + Vocabulary Mapping

In sequence modeling (LSTMs, autoregressive Transformers), tokenization and vocabulary mapping follow the same principle as NLP — but the *unit of token* changes dramatically depending on what you're modeling.

---

## 1. What Are We Tokenizing?

| Data Type | Token Unit | Example |
|---|---|---|
| Text (char-level) | Single character | `"h","e","l","l","o"` |
| Text (word-level) | Word | `"hello","world"` |
| Text (subword) | BPE / WordPiece piece | `"hel","lo"` |
| MIDI Music | Note event | `NOTE_ON(60), NOTE_OFF(60)` |
| ABC Notation | Symbol | `G`, `A2`, `|` |
| Code | Token | `def`, `(`, `x`, `:` |
| DNA | Nucleotide | `A`, `T`, `G`, `C` |

---

## 2. Character-Level Tokenization (Most Common for LSTMs)

Character-level is the standard for sequence modeling with LSTMs — zero OOV problem, tiny vocabulary, works on any language or domain.

```python
text = open("shakespeare.txt", "r").read()

# Build vocabulary — all unique characters
vocab = sorted(set(text))
print(f"Vocab size: {len(vocab)}")   # typically 60–100 chars

# Mappings
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char  = {i: ch for i, ch in enumerate(vocab)}

# Encode entire text
encoded = [char2idx[ch] for ch in text]
print(encoded[:10])   # [18, 47, 56, 57, 58, ...]
```

---

## 3. Word-Level Tokenization

Used when you want the model to reason at the word level — language modeling, text generation.

```python
import re
from collections import Counter

text = open("corpus.txt").read().lower()
tokens = re.findall(r"\b\w+\b", text)

# Build vocab with frequency threshold
freq = Counter(tokens)
min_freq = 2

special = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
vocab_words = [w for w, c in freq.most_common() if c >= min_freq]
vocab = special + vocab_words

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

UNK = word2idx["<UNK>"]
encoded = [word2idx.get(tok, UNK) for tok in tokens]
```

---

## 4. Music Tokenization (MIDI)

Music is tokenized as a sequence of events — note on, note off, time shift, velocity.

```python
# MIDI event vocabulary example
VOCAB = {
    "NOTE_ON_60":   0,    # Middle C on
    "NOTE_ON_64":   1,    # E on
    "NOTE_ON_67":   2,    # G on
    "NOTE_OFF_60":  3,
    "NOTE_OFF_64":  4,
    "NOTE_OFF_67":  5,
    "TIME_SHIFT_10": 6,   # advance 10ms
    "VELOCITY_80":  7,
    "<PAD>":        8,
    "<EOS>":        9,
}

# Each MIDI file → sequence of integer event IDs
# "play C major chord" → [7, 0, 1, 2, 6, 3, 4, 5]
```

Popular MIDI tokenization libraries: `miditok`, `pretty_midi`, `music21`

---

## 5. Special Tokens for Sequence Modeling

| Token | Purpose |
|---|---|
| `<PAD>` (0) | Padding to uniform length |
| `<UNK>` (1) | Unknown / OOV token |
| `<BOS>` | Beginning of sequence — seed for generation |
| `<EOS>` | End of sequence — stop signal during inference |

```python
# Wrap sequences with BOS and EOS
def encode_sequence(text, char2idx):
    BOS = char2idx["<BOS>"]
    EOS = char2idx["<EOS>"]
    return [BOS] + [char2idx[ch] for ch in text] + [EOS]
```

---

## 6. Vocabulary Size Guidelines

| Domain | Tokenization | Typical Vocab Size |
|---|---|---|
| Text (char) | Character | 60–130 |
| Text (word) | Word | 10K–100K |
| Text (subword) | BPE | 30K–50K |
| MIDI music | Event-based | 300–500 |
| DNA | Nucleotide | 4–8 |
| Code | Subword | 32K–50K |

Smaller vocabulary → simpler embedding table, faster training, but model must compose meaning from smaller units.