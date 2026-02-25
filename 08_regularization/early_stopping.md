# Early Stopping — Regularization in the Training Loop

---

## What Is Early Stopping?

Every other regularization technique we discuss modifies the model or the optimizer. Early stopping is unique: it **regulates the training process itself** by deciding when to stop.

The problem it solves is fundamental: a neural network, given enough training time, will eventually memorize its training data. The training loss will keep decreasing long after the model has stopped learning generalizable patterns — it's just memorizing noise and specific examples. Validation performance peaks and then starts to degrade.

Early stopping monitors validation performance and **halts training at the point of best generalization** — before the model has overfit.

```
Training loss:     continuously ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
Validation loss:   ↓ ↓ ↓ ↓ [BEST] → ↑ ↑ ↑ ↑ ↑ ↑ ↑

                                ↑
                         Stop here, not at the end of training
```

The key insight from Occam's razor applied to training: **the longer you train, the more complex the hypothesis your model fits**. Early stopping limits this complexity by constraining the number of gradient steps — and in this sense it's analogous to weight decay. Both limit model complexity, just through different mechanisms.

---

## The Overfitting Timeline

Understanding when to stop requires understanding the typical shape of training dynamics:

**Phase 1 — Rapid learning**: Both training and validation loss drop quickly. The model is learning genuinely useful patterns. Keep training.

**Phase 2 — Refinement**: Training loss still decreasing, validation loss decreasing more slowly. The model is refining its general representations. Keep training.

**Phase 3 — Divergence begins**: Training loss still decreasing, validation loss flat or slightly increasing. The model is beginning to memorize training-specific noise. This is the danger zone.

**Phase 4 — Overfitting**: Training loss low, validation loss clearly increasing. The model has overfit. You should have stopped earlier.

Early stopping is the mechanism that catches you between Phase 2 and Phase 3.

---

## Core Concepts

### Patience

The most important hyperparameter: how many epochs to wait **without improvement** before stopping.

Setting patience to 1 means "stop the moment validation loss goes up once" — this is too aggressive. Validation loss is noisy; it will fluctuate up on some epochs even when the trend is still improving. You'd stop far too early.

Setting patience to 100 means you're willing to wait 100 epochs of degradation before stopping — this defeats the purpose.

**Practical guidance:**

| Dataset/Task | Typical Patience |
|-------------|-----------------|
| Small dataset (<10k samples) | 5–10 epochs |
| Medium dataset (10k–100k) | 10–20 epochs |
| Large dataset (>100k) | 20–50 epochs |
| Fine-tuning pretrained models | 3–7 epochs |
| LR scheduled training (cosine) | Often not used — fixed epochs instead |

Patience should be larger when:
- Validation loss is noisy (small dataset, small validation set)
- You're using learning rate warmup (loss can temporarily increase during warmup)
- The model is large and training is slow

### The Delta Threshold

"Improvement" should often be defined with a minimum threshold (delta). A validation loss improvement of `0.00001` might be within the noise floor of your dataset and not meaningful.

```python
# min_delta: minimum change to count as improvement
if (best_loss - current_loss) > min_delta:
    improvement = True
```

Typical `min_delta` values: `0.001` for loss, `0.1%` for accuracy.

### Save the Best Checkpoint

Early stopping doesn't just stop training — it requires saving the model weights at the point of best validation performance. Training continues until patience runs out, but you restore to the saved checkpoint at the end. Without checkpoint saving, you'd restore the model from the overfitted state at the final epoch.

```
Epoch 45: val_loss = 0.182  ← NEW BEST → save checkpoint
Epoch 46: val_loss = 0.185  ← no improvement (1/patience)
Epoch 47: val_loss = 0.183  ← no improvement (2/patience)
Epoch 48: val_loss = 0.190  ← no improvement (3/patience) → STOP
→ Load checkpoint from epoch 45
```

---

## Implementation ⭐

### From Scratch in PyTorch

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss, model):
        if (self.best_loss - val_loss) > self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                # Save a deep copy of the best weights
                import copy
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            # No improvement
            self.counter += 1
        
        if self.counter >= self.patience:
            # Patience exhausted — stop training
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True   # Signal to stop
        
        return False      # Signal to continue


# Training loop integration
early_stopping = EarlyStopping(patience=15, min_delta=0.001)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, optimizer, train_loader)
    val_loss = evaluate(model, val_loader)
    
    scheduler.step(val_loss)   # If using ReduceLROnPlateau
    
    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    if early_stopping(val_loss, model):
        print(f"Early stopping at epoch {epoch}. Best val loss: {early_stopping.best_loss:.4f}")
        break
```

### Monitoring Accuracy Instead of Loss

Sometimes validation loss isn't the right metric — you care about accuracy, F1, or AUC. The patience logic flips direction:

```python
class EarlyStopping:
    def __init__(self, patience=10, mode='min', min_delta=0.001):
        self.patience = patience
        self.mode = mode  # 'min' for loss, 'max' for accuracy/AUC
        self.min_delta = min_delta
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0

    def __call__(self, score, model):
        if self.mode == 'min':
            improved = (self.best_score - score) > self.min_delta
        else:
            improved = (score - self.best_score) > self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            # save checkpoint...
        else:
            self.counter += 1
        
        return self.counter >= self.patience
```

### Using PyTorch Lightning (Automatic Early Stopping)

PyTorch Lightning makes this trivial:

```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_loss',    # Metric to watch
    patience=10,
    mode='min',
    min_delta=0.001,
    verbose=True
)

checkpoint = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,          # Keep only the best checkpoint
    mode='min',
    filename='best-{epoch}-{val_loss:.4f}'
)

trainer = pl.Trainer(
    max_epochs=500,
    callbacks=[early_stop, checkpoint]
)
```

---

## Early Stopping vs. Learning Rate Schedules

A common confusion: if you're using a fixed-epoch scheduler like CosineAnnealingLR, should you use early stopping?

**The tension:** CosineAnnealingLR is designed for a specific number of epochs. If early stopping fires at epoch 60 of a 200-epoch schedule, the LR is still high (mid-cosine) and you may not have converged to the best point.

**In practice:**

| Training Setup | Use Early Stopping? |
|---------------|-------------------|
| Fixed epochs + CosineAnnealingLR | Rarely — trust the schedule |
| Fixed epochs + StepLR | Sometimes — as a safety net |
| ReduceLROnPlateau (adaptive) | Yes — designed to pair with it |
| Fine-tuning (variable duration) | Yes — standard practice |
| Training from scratch (large models) | Rarely — use fixed epochs + schedule |

For transformer fine-tuning (BERT, GPT), early stopping with patience 3–5 is standard because fine-tuning is prone to overfitting quickly on small datasets.

---

## What Metric to Monitor?

Choosing the right validation metric is critical:

**Validation Loss** — most common, directly tied to the optimization objective. Can be noisy on small validation sets.

**Task Metric (Accuracy, F1, AUC)** — what you actually care about. Can be less noisy than loss for classification tasks. Use this for imbalanced datasets where accuracy is misleading — monitor F1 or AUC instead.

**Rule:** Monitor the metric that aligns with your deployment goal. If you care about accuracy at deployment, monitor accuracy. If loss is your evaluation criterion, monitor loss.

---

## Common Pitfalls

**Using the test set for early stopping** — this leaks test information into training decisions. The test set must be untouched until final evaluation. Early stopping must use a held-out **validation set**.

**Validation set too small** — a small validation set produces noisy metrics, causing false triggers. If your dataset is small, consider k-fold cross-validation with early stopping, or use a larger validation fraction.

**Patience too low + noisy validation** — validation loss naturally fluctuates. With patience=3 and noisy data, you'll stop at a suboptimal point. Increase patience or use running average smoothing:

```python
# Smooth val loss with exponential moving average before checking patience
smooth_val_loss = 0.9 * smooth_val_loss + 0.1 * current_val_loss
early_stopping(smooth_val_loss, model)
```

**Forgetting to restore weights** — stopping without loading the best checkpoint means you end up with the overfitted model from the last epoch, not the best one.

---

## Early Stopping as a Complexity Regulator

A theoretical perspective worth understanding: early stopping and L2 regularization (weight decay) are closely related. Both limit model complexity — weight decay by keeping weights small, early stopping by limiting the number of gradient steps.

For linear models, early stopping with gradient descent is mathematically equivalent to L2 regularization with a specific `λ` that depends on the stopping time. For nonlinear networks, the relationship is more complex but the intuition holds: **fewer gradient steps = simpler function**.

This means early stopping and weight decay are **partially redundant** as regularizers. Using both provides complementary protection, but either alone can be sufficient depending on the task.

---

## Key Takeaways

- Early stopping is **the simplest form of regularization** — it doesn't require modifying the model or optimizer, just the training loop.
- **Always save and restore the best checkpoint** — stopping without restoring weights discards the entire purpose.
- **Patience is the key hyperparameter** — too low causes premature stopping, too high defeats the purpose. Start with patience=10 and adjust based on validation noise.
- **Pair with ReduceLROnPlateau** for adaptive training, or use fixed-epoch training with cosine scheduling if you know your training duration.
- For fine-tuning pretrained models, early stopping with **patience=3–7** is standard practice and often the primary regularization mechanism.
- Monitor the metric you actually care about at deployment, not just training loss.