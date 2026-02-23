# Sequence Windowing + Input / Target Pairs

This is the core step that transforms a flat encoded sequence into a dataset the model can learn from. The model is trained to predict the *next token* given all previous tokens — this is called **autoregressive** or **language modeling** objective.

---

## 1. The Core Idea

Given the sequence `"hello"`:

```
Input  →  Target  (model must predict target given input)

"h"    →  "e"
"he"   →  "l"
"hel"  →  "l"
"hell" →  "o"
```

More practically with a fixed window size `seq_len = 4`:

```
Encoded:  [8, 5, 12, 12, 15, 23, 15, 18, 12, 4]
           h   e   l   l   o   w   o   r   l   d

Window 1:  Input=[8, 5, 12, 12]   Target=[5, 12, 12, 15]
Window 2:  Input=[5, 12, 12, 15]  Target=[12, 12, 15, 23]
Window 3:  Input=[12, 12, 15, 23] Target=[12, 15, 23, 15]
...

Target is always Input shifted by 1 position to the right.
```

---

## 2. Sliding Window (Overlapping)

Each window overlaps with the previous by `(seq_len - 1)` positions — creates the maximum number of training samples.

```python
import numpy as np

def create_sequences(encoded, seq_len):
    inputs  = []
    targets = []
    for i in range(len(encoded) - seq_len):
        inputs.append(encoded[i : i + seq_len])
        targets.append(encoded[i + 1 : i + seq_len + 1])
    return np.array(inputs), np.array(targets)

encoded = [8, 5, 12, 12, 15, 23, 15, 18, 12, 4]
seq_len = 4

X, y = create_sequences(encoded, seq_len)
print(X.shape)   # (6, 4)  →  6 windows of length 4
print(y.shape)   # (6, 4)

print(X[0])   # [8, 5, 12, 12]
print(y[0])   # [5, 12, 12, 15]
```

---

## 3. Non-Overlapping Windows (Stride = seq_len)

Faster and uses less memory — windows don't overlap. Fewer training samples but much more efficient.

```python
def create_sequences_nonoverlap(encoded, seq_len):
    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_len, seq_len):  # stride = seq_len
        inputs.append(encoded[i : i + seq_len])
        targets.append(encoded[i + 1 : i + seq_len + 1])
    return np.array(inputs), np.array(targets)
```

**Overlapping vs Non-Overlapping:**

| | Overlapping | Non-Overlapping |
|---|---|---|
| Samples generated | `N - seq_len` | `N // seq_len` |
| Redundancy | High | None |
| Training data size | Large | Small |
| Common for | Character LSTMs, small datasets | Large corpora, Transformer pretraining |

---

## 4. PyTorch Dataset Implementation

```python
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, encoded, seq_len):
        self.encoded = torch.LongTensor(encoded)
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        x = self.encoded[idx     : idx + self.seq_len]
        y = self.encoded[idx + 1 : idx + self.seq_len + 1]
        return x, y

dataset = SequenceDataset(encoded, seq_len=64)
print(f"Total windows: {len(dataset)}")

x, y = dataset[0]
print(x.shape)   # torch.Size([64])
print(y.shape)   # torch.Size([64])
```

---

## 5. Choosing seq_len

`seq_len` defines how far back the model can see when making a prediction — its **context window**.

| Model | Typical seq_len |
|---|---|
| Character LSTM (small) | 64–128 |
| Character LSTM (large) | 256–512 |
| Word-level LSTM | 35–100 |
| GPT-2 | 1024 |
| GPT-3 | 2048 |
| LLaMA 2 | 4096 |
| LLaMA 3 | 8192 |

Longer `seq_len`:
- Model can learn longer-range dependencies
- More memory per batch
- Requires more training data to see full benefit

---

## 6. Truncated Backpropagation Through Time (TBPTT)

For very long sequences, computing gradients all the way back through the full sequence is expensive and causes vanishing gradients. TBPTT breaks the sequence into fixed chunks and detaches the hidden state between chunks.

```python
seq_len   = 35    # backprop window (how far back gradients flow)
chunk_len = 35    # each forward pass processes this many steps

hidden = None

for i in range(0, len(data) - seq_len, chunk_len):
    x_chunk = data[i : i + chunk_len]
    y_chunk = data[i + 1 : i + chunk_len + 1]

    # Detach hidden state — stop gradient flow between chunks
    if hidden is not None:
        hidden = tuple(h.detach() for h in hidden)

    output, hidden = model(x_chunk, hidden)
    loss = criterion(output, y_chunk)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 7. Transformer: Causal Masking Instead of Windowing

Transformers process the entire sequence at once, but use a **causal mask** to prevent each position from attending to future tokens — achieving the same autoregressive property without explicit windowing.

```python
import torch

def causal_mask(seq_len):
    # Upper triangular mask — future positions are -inf (ignored by softmax)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask

mask = causal_mask(5)
# tensor([[0., -inf, -inf, -inf, -inf],
#         [0.,  0.,  -inf, -inf, -inf],
#         [0.,  0.,   0.,  -inf, -inf],
#         [0.,  0.,   0.,   0.,  -inf],
#         [0.,  0.,   0.,   0.,   0.]])
```

Position `i` can only attend to positions `0` through `i` — never the future.