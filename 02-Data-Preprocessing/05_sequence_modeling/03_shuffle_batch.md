# Shuffle + Batch

The final steps before data enters the model. Shuffling prevents the model from memorizing order. Batching groups samples for efficient parallel computation on GPU.

---

## 1. Why Shuffle?

Without shuffling, the model sees training samples in the same order every epoch. This causes:
- The model learns the ordering pattern instead of the data patterns
- Gradient updates become correlated — SGD loses its stochastic benefit
- Training loss may appear smooth but generalization is poor

```python
# Without shuffle: epoch 1 sees [A,B,C,D], epoch 2 sees [A,B,C,D]
# With shuffle:    epoch 1 sees [C,A,D,B], epoch 2 sees [B,D,A,C]
```

**Exception — do NOT shuffle:**
- Time series where sequence order within a batch matters
- Stateful LSTMs where hidden state must carry over between consecutive batches
- When using TBPTT with ordered chunks

---

## 2. PyTorch DataLoader — Shuffle + Batch in One

```python
import torch
from torch.utils.data import DataLoader

# Shuffle automatically handled by DataLoader
train_loader = DataLoader(
    dataset,
    batch_size  = 64,      # number of sequences per batch
    shuffle     = True,    # shuffle at start of every epoch
    num_workers = 4,       # parallel data loading processes
    pin_memory  = True,    # faster CPU→GPU transfer
    drop_last   = True,    # drop final incomplete batch (keeps batch size uniform)
)

# Iteration
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)   # (B, seq_len)
        y_batch = y_batch.to(device)   # (B, seq_len)
        output  = model(x_batch)
        loss    = criterion(output.view(-1, vocab_size), y_batch.view(-1))
```

---

## 3. Choosing Batch Size

| Batch Size | Effect |
|---|---|
| Small (8–32) | Noisy gradients → acts as regularization · slower per epoch |
| Medium (64–256) | Standard trade-off for most sequence models |
| Large (512–2048) | Stable gradients · faster epochs · needs higher LR · needs more GPU memory |

**Memory estimation:**
```
Memory per batch ≈ batch_size × seq_len × embedding_dim × 4 bytes
Example: 64 × 128 × 256 × 4 = 8 MB per batch
```

**Rule of thumb:** Start with `batch_size=64`, scale up until GPU memory is ~80% utilized.

---

## 4. Manual Shuffle + Batch (Without DataLoader)

```python
import numpy as np

def create_batches(X, y, batch_size, shuffle=True):
    n = len(X)
    if shuffle:
        indices = np.random.permutation(n)
        X, y = X[indices], y[indices]

    for start in range(0, n - batch_size + 1, batch_size):
        yield (
            torch.LongTensor(X[start : start + batch_size]),
            torch.LongTensor(y[start : start + batch_size])
        )

# Usage
for x_batch, y_batch in create_batches(X, y, batch_size=64):
    output = model(x_batch.to(device))
```

---

## 5. Stateful Batching for LSTMs

For character-level LSTMs, you often want the hidden state to carry over between consecutive batches — the model remembers what it saw earlier in the text.

```python
# Split data into B parallel streams — each stream feeds one position in the batch
def stateful_batches(encoded, batch_size, seq_len):
    n = len(encoded)
    chunk = n // batch_size

    # Reshape into (batch_size, chunk)
    data = torch.LongTensor(encoded[:batch_size * chunk])
    data = data.view(batch_size, chunk)

    for i in range(0, chunk - seq_len, seq_len):
        x = data[:, i     : i + seq_len]    # (batch_size, seq_len)
        y = data[:, i + 1 : i + seq_len + 1]
        yield x, y

# Hidden state is passed between batches — NOT detached between chunks
# (unless using TBPTT — then detach every N steps)
```

---

## 6. Gradient Accumulation (Large Effective Batch)

If GPU memory limits you to small batches but you want the effect of a larger batch:

```python
accumulation_steps = 4   # effective batch = batch_size × 4

optimizer.zero_grad()
for step, (x, y) in enumerate(train_loader):
    output = model(x.to(device))
    loss   = criterion(output.view(-1, vocab_size), y.view(-1).to(device))
    loss   = loss / accumulation_steps   # scale loss
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 7. Gradient Clipping

Sequence models (especially LSTMs) are prone to **exploding gradients** — always clip before optimizer step.

```python
# Clip gradient norm to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

---

## 8. Complete Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model     = MyLSTM(vocab_size=65, embed_dim=128, hidden_dim=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(dataset, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)   # (B, seq_len)
        y_batch = y_batch.to(device)   # (B, seq_len)

        optimizer.zero_grad()
        output  = model(x_batch)       # (B, seq_len, vocab_size)

        loss = criterion(
            output.view(-1, vocab_size),  # (B*seq_len, vocab_size)
            y_batch.view(-1)              # (B*seq_len,)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    perplexity = np.exp(avg_loss)   # standard metric for language models
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
```

---

## 9. Key Metrics for Sequence Models

| Metric | Formula | Meaning |
|---|---|---|
| Cross-Entropy Loss | `-log P(next token)` | How surprised the model is by the correct token |
| Perplexity | `exp(loss)` | Effective vocabulary size the model is choosing from |
| Bits per character (BPC) | `loss / log(2)` | For character-level models |

Lower perplexity = better model. Random guess on 65-char vocab → perplexity = 65.