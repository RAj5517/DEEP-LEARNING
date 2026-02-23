# Embedding Layer

The embedding layer is the **bridge between discrete token indices and continuous vector space**. It converts integer token IDs into dense floating-point vectors that the model can compute with.

---

## 1. What is an Embedding?

After vocabulary mapping, each token is an integer (e.g., `"hello" → 42`). But an integer carries no semantic information — `42` and `43` are just numbers. An embedding converts each index into a learned dense vector.

```
Token index: 42
     ↓
Embedding lookup
     ↓
Dense vector: [0.23, -0.11, 0.87, 0.04, -0.56, ...]  ← shape: (embedding_dim,)
```

The embedding matrix has shape `(vocab_size, embedding_dim)`. Each row is the vector for one token. These vectors are **learned during training** — the model adjusts them to capture meaning.

---

## 2. Implementation in PyTorch

```python
import torch
import torch.nn as nn

vocab_size    = 10000   # number of unique tokens
embedding_dim = 256     # size of each token vector
pad_idx       = 0       # padding token index — its vector stays zero

embedding = nn.Embedding(
    num_embeddings = vocab_size,
    embedding_dim  = embedding_dim,
    padding_idx    = pad_idx       # gradient is zeroed for PAD token
)

# Forward pass
token_ids = torch.LongTensor([[4, 7, 2, 0, 0]])   # shape: (batch=1, seq_len=5)
vectors   = embedding(token_ids)                    # shape: (1, 5, 256)

print(vectors.shape)  # torch.Size([1, 5, 256])
```

---

## 3. Implementation in Keras / TensorFlow

```python
from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(
    input_dim    = 10000,   # vocab size
    output_dim   = 256,     # embedding dimension
    mask_zero    = True,    # automatically masks padding (index 0)
    input_length = 128      # fixed sequence length (optional)
)

# In a model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    Embedding(10000, 256, input_length=128),
    LSTM(128),
    Dense(1, activation="sigmoid")
])
```

---

## 4. Choosing Embedding Dimension

The embedding dimension controls how much information each token can encode.

| Vocab Size | Recommended Embedding Dim |
|---|---|
| < 1,000 | 32–64 |
| 1K – 10K | 64–128 |
| 10K – 50K | 128–256 |
| 50K – 100K | 256–512 |
| LLMs (GPT, BERT) | 768–12288 |

Rule of thumb: `embedding_dim = min(512, vocab_size ** 0.25 * 16)`

---

## 5. Pretrained Embeddings (Transfer Learning)

Instead of learning embeddings from scratch, initialize with vectors pretrained on large corpora. The model starts with semantic knowledge already baked in.

### 5.1 Word2Vec / GloVe / FastText (Static)

Fixed vectors — they don't change based on context.

```python
import gensim.downloader as api
import numpy as np

# Load pretrained GloVe vectors
glove = api.load("glove-wiki-gigaword-100")  # 100-dim GloVe

# Build embedding matrix aligned to your vocabulary
embedding_matrix = np.zeros((vocab_size, 100))
for token, idx in vocab.items():
    if token in glove:
        embedding_matrix[idx] = glove[token]

# Load into PyTorch embedding layer
embedding = nn.Embedding(vocab_size, 100)
embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))

# Option 1: Freeze (don't update during training)
embedding.weight.requires_grad = False

# Option 2: Fine-tune (allow updates)
embedding.weight.requires_grad = True
```

### 5.2 Contextual Embeddings (BERT, RoBERTa, LLaMA)

Modern transformers produce **context-dependent** embeddings — the same word gets a different vector depending on its surrounding context.

```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("The bank by the river", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: shape (batch, seq_len, 768)
token_embeddings = outputs.last_hidden_state
print(token_embeddings.shape)  # torch.Size([1, 7, 768])
```

`"bank"` near `"river"` gets a different vector than `"bank"` near `"money"` — this is the key advantage over static embeddings.

---

## 6. Positional Encoding (Transformers)

Transformers process all tokens in parallel — they have no built-in notion of order. Positional encodings are **added** to token embeddings to inject position information.

### 6.1 Sinusoidal (Original Transformer)

```python
import torch
import math

def sinusoidal_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape: (seq_len, d_model)

# In forward pass:
# x = token_embedding + positional_encoding
```

### 6.2 Learned Positional Embeddings (BERT, GPT)

```python
# BERT-style: position is just another embedding lookup
position_embedding = nn.Embedding(max_seq_len, embedding_dim)
positions = torch.arange(seq_len).unsqueeze(0)  # shape (1, seq_len)

combined = token_embedding(token_ids) + position_embedding(positions)
```

---

## 7. Embedding Visualization

Embeddings encode semantic relationships. Similar words cluster together in vector space.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ["king", "queen", "man", "woman", "paris", "france", "london", "england"]
vectors = np.array([glove[w] for w in words])

pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    plt.annotate(word, reduced[i])
plt.title("Word Embeddings (PCA)")
plt.show()

# Famous relationship: king - man + woman ≈ queen
result = glove["king"] - glove["man"] + glove["woman"]
glove.similar_by_vector(result, topn=3)
# → [('queen', 0.85), ...]
```

---

## 8. Static vs Contextual Embeddings

| | Static (Word2Vec, GloVe) | Contextual (BERT, GPT) |
|---|---|---|
| Same word, different context | Same vector | Different vector |
| OOV handling | Requires subword tricks | Handled by tokenizer |
| Speed | Fast | Slower |
| Quality | Good baseline | State-of-the-art |
| Use case | Traditional NLP, lightweight | Transformers, fine-tuning |

---

## 9. Full Pipeline Summary

```
Raw text
   ↓
Tokenizer   →   ["hello", "world"]
   ↓
Vocab map   →   [42, 87]
   ↓
Padding     →   [42, 87, 0, 0]
   ↓
Embedding   →   shape (batch, seq_len, embedding_dim)
   ↓
LSTM / Transformer / LLM
```