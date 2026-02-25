# Mixup & CutMix — Advanced Augmentation

---

## What Are These Techniques?

Standard augmentations (flips, crops, color jitter) transform a single image while preserving its label. Mixup and CutMix do something fundamentally different: they **combine two training examples** — both the inputs and the labels — creating entirely new synthetic training samples.

This category of augmentation is built on a key idea: real-world data lives on a continuous manifold, not as discrete isolated points. If the model sees only clean, well-separated examples at training time, it may behave unexpectedly in the space *between* training examples — exactly where ambiguous real-world inputs tend to fall.

By training on **linear interpolations** (Mixup) or **spatial composites** (CutMix) of pairs of examples, you force the model to behave smoothly and predictably across the entire input space, not just at the training points. This is a form of **vicinal risk minimization** — instead of minimizing risk only at exact training points, you minimize it in a neighborhood around each point.

The result is typically:
- Better generalization (the model can't memorize exact training images)
- Smoother decision boundaries
- Better calibration (model uncertainty is more meaningful)
- Significant accuracy improvements, especially when combined with longer training schedules

---

## Mixup ⭐ Important

### The Idea

Take two random training examples `(x₁, y₁)` and `(x₂, y₂)`. Create a new training example by **pixel-wise linear interpolation** of both the images and the labels:

```
λ ~ Beta(α, α)        # Sample mixing coefficient
x̃ = λ · x₁ + (1-λ) · x₂    # Mix images
ỹ = λ · y₁ + (1-λ) · y₂    # Mix labels (soft targets)
```

The Beta distribution with parameter `α` controls how much mixing happens:
- `α → 0`: λ ≈ 0 or λ ≈ 1 most of the time → almost no mixing → original images
- `α = 1`: λ ~ Uniform(0, 1) → full range of mixing ratios
- `α > 1`: λ concentrates near 0.5 → heavy mixing, blended images dominate

Standard value: `α = 0.2` (mild mixing) to `α = 1.0` (more aggressive mixing).

### What the Mixed Images Look Like

```
Original x₁: [cat image]      y₁ = [1, 0, 0] (cat)
Original x₂: [dog image]      y₂ = [0, 1, 0] (dog)

With λ = 0.7:
Mixed x̃: [70% cat + 30% dog, superimposed]
Mixed ỹ: [0.7, 0.3, 0.0]    (70% cat, 30% dog)
```

The model sees a ghostly superimposition of two images and is trained to predict a weighted combination of both labels. This is a strange-looking image that would never appear in the real world — and that's fine. The model doesn't need to recognize mixed images at test time; it just needs to learn smooth, interpolable representations.

### Why This Works

The key insight is in the loss function. For a cross-entropy loss with mixed labels:

```
L = λ · CE(f(x̃), y₁) + (1-λ) · CE(f(x̃), y₂)
```

The model is penalized for ignoring either class in the mixed image. This means:
1. Decision boundaries must be smooth — there's no hard boundary between cat and dog
2. The model must learn representations where "60% cat, 40% dog" produces appropriate interpolated predictions
3. Memorization is prevented — memorizing specific pixel patterns is impossible when every image is a blend

### Implementation

```python
def mixup_data(x, y, alpha=0.2):
    """Apply mixup to a batch of images and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    # Random permutation of the batch to get pairs
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# In training loop:
criterion = nn.CrossEntropyLoss()

for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    
    # Apply mixup
    mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
    
    outputs = model(mixed_images)
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Using torchvision (Modern)

```python
from torchvision.transforms import v2

# As a transform in the pipeline
mixup = v2.MixUp(alpha=0.2, num_classes=1000)

for images, labels in train_loader:
    images, labels = mixup(images, labels)   # Returns mixed batch
    # labels are now soft: shape [batch, num_classes]
    
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)  # Works with soft labels
```

### Hyperparameter Guide

| α | Mixing behavior | Typical use |
|---|----------------|------------|
| 0.1 | Mild — mostly original images | When you want subtle regularization |
| **0.2** | **Standard — moderate mixing** | **Default for ImageNet, most CV** |
| 0.4 | Moderate-aggressive | When dataset is small or noisy |
| 1.0 | Fully uniform mixing ratio | When combining with other augmentations |

### Mixup in NLP

Mixup works in embedding space for NLP — mix embeddings rather than raw token ids. This is called **EmbedMix** and is used in text classification. For sequence models with attention, mixing in the hidden state space (instead of input space) is more common.

---

## CutMix ⭐ Important

### The Idea

CutMix (Yun et al., 2019, ICCV) is a natural evolution of Mixup that fixes a key limitation: Mixup creates unnatural "ghost" superimpositions that don't look like any real image. CutMix instead **cuts a rectangular patch from one image and pastes it into another**:

```
x₁: [full cat image]
x₂: [full dog image]

CutMix: [cat image with a rectangular dog patch cut in]
ỹ = λ · y₁ + (1-λ) · y₂   where λ = fraction of image from x₁
```

The mixed label is proportional to the **pixel area** from each source image. If the dog patch covers 30% of the image, the label is 70% cat + 30% dog.

### The Key Advantage Over Mixup

CutMix generates images that are locally consistent — each patch is a real, coherent portion of an image. The model must recognize objects from **partial views** with clear spatial boundaries. This teaches:

1. **Local feature recognition**: The model can't rely on global image statistics — it must identify discriminative local features
2. **Spatial robustness**: Objects can be occluded by other objects (realistic!) and the model must still classify correctly
3. **Better feature localization**: Because the model must identify where meaningful content is, it implicitly learns to attend to the right regions — CutMix models often produce better **class activation maps** (CAMs)

### Implementation

```python
def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix to a batch."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    _, _, H, W = x.size()
    
    # Sample the cut box
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    # Random box center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Clamp to image boundaries
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Paste the patch
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # Recalculate lambda based on actual patch area
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Training loop — same as Mixup:
mixed_images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
outputs = model(mixed_images)
loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
```

### Using torchvision (Modern)

```python
from torchvision.transforms import v2

cutmix = v2.CutMix(alpha=1.0, num_classes=1000)

for images, labels in train_loader:
    images, labels = cutmix(images, labels)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
```

---

## Mixup vs. CutMix: When to Use Which?

Both improve generalization, but they do so through different mechanisms with different strengths:

| Property | Mixup | CutMix |
|----------|-------|--------|
| Image appearance | Ghostly superimposition | Natural local patches |
| What it teaches | Global interpolation | Local feature recognition |
| Classification accuracy | Good | Often slightly better |
| Object detection | Awkward | Much more natural |
| Segmentation | Difficult to use | More compatible |
| Feature localization (CAM quality) | Slightly worse | Significantly better |
| Implementation complexity | Simple | Slightly more complex |

**Use Mixup when:**
- Pure image classification
- You want maximum simplicity
- NLP or other non-spatial domains

**Use CutMix when:**
- Image classification where you want sharper CAMs / better localization
- Combined training on classification + detection/segmentation
- When objects have discriminative local features (fine-grained classification)

### Combining Both: CutMix + Mixup

A popular strategy: apply CutMix with probability 0.5 and Mixup with probability 0.5 at each batch. This gets the benefits of both:

```python
from torchvision.transforms import v2

# Apply one or the other randomly
cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
mixup = v2.MixUp(alpha=0.2, num_classes=num_classes)

def apply_augmentation(images, labels):
    if np.random.rand() < 0.5:
        return cutmix(images, labels)
    else:
        return mixup(images, labels)
```

This is used in **DeiT** (the original vision transformer training recipe) and is part of the `timm` default training configuration.

---

## The Modern Training Recipe

For training a strong image classifier (ResNet, ViT, ConvNeXt) from scratch on ImageNet, the current state-of-the-art training recipe combines everything:

```python
from torchvision.transforms import v2

# Augmentation pipeline
train_transform = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.TrivialAugmentWide(),         # Or RandAugment(2, 9)
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.25),
])

# Batch-level augmentation (applied after batching)
cutmix = v2.CutMix(alpha=1.0, num_classes=1000)
mixup = v2.MixUp(alpha=0.2, num_classes=1000)

# Training loop
for images, labels in train_loader:
    # Apply CutMix or Mixup randomly
    if np.random.rand() < 0.5:
        images, labels = cutmix(images, labels)
    else:
        images, labels = mixup(images, labels)
    
    outputs = model(images)
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, labels)
    # ↑ Note: label smoothing + mixup/cutmix work well together
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

This recipe is used in `timm` (PyTorch Image Models) and is the basis for training results in DeiT, ConvNeXt, and other modern architectures.

---

## FMix and SnapMix: Brief Introductions

**FMix** (Harris et al., 2020): Instead of rectangular patches (CutMix) or full superimposition (Mixup), FMix creates irregular, Fourier-space masks — organic blob-shaped mix regions. Slight improvements over CutMix in some settings, but the complexity rarely justifies the marginal gain.

**SnapMix** (Huang et al., 2021): Makes CutMix semantically-aware by using class activation maps to guide where patches are cut. More likely to cut from semantically meaningful regions. Useful for fine-grained recognition where discriminative regions are small and specific.

**SamplePairing** (Inoue, 2018): A simpler precursor to Mixup that averages exactly two images without the Beta distribution — effectively Mixup with fixed `λ = 0.5`. Less flexible and largely superseded by Mixup.

---

## Key Takeaways

- **Mixup and CutMix are among the highest-impact improvements** to standard training — both consistently improve ImageNet accuracy by 0.5–2% and generalization across benchmarks.
- **Mixup** (α=0.2) is simpler and works across domains (vision, NLP embeddings, audio). Start here.
- **CutMix** (α=1.0) produces more natural mixed images and better feature localization. Often slightly better than Mixup for pure classification.
- **Combine both** (random 50/50 per batch) to get the benefits of both — this is the modern DeiT/timm standard.
- Both require **soft label handling** in the loss function — make sure your loss function accepts soft targets or use the two-term formulation `λ·CE(y_a) + (1-λ)·CE(y_b)`.
- **Label smoothing stacks well** with Mixup/CutMix — they address different aspects of overconfidence.
- CutMix produces naturally better **class activation maps** — consider it when interpretability or localization quality matters alongside accuracy.