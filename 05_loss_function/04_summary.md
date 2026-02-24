## Quick Reference Table

| Loss Function | Category | Use Case | When to Use | PyTorch |
|--------------|----------|----------|-------------|---------|
| **MSE** | Regression | General regression | Default, well-behaved data | `nn.MSELoss()` |
| **Cross Entropy** | Classification | Multi-class | 3+ mutually exclusive classes | `nn.CrossEntropyLoss()` |
| **BCE** | Classification | Binary/Multi-label | 2 classes or independent labels | `nn.BCEWithLogitsLoss()` |
| **Focal Loss** | Classification | Imbalanced | Severe class imbalance | Custom |
| **Dice Loss** | Segmentation | Medical imaging | Small objects, pixel imbalance | Custom |
| **InfoNCE** | Metric Learning | Embeddings | Self-supervised, CLIP-style | Custom |
| **KL Divergence** | Distribution | VAE, Distillation | Match distributions | `nn.KLDivLoss()` |
| **Wasserstein** | GAN | Image generation | Stable GAN training | Custom (WGAN-GP) |
| **Smooth L1** | Regression | Object detection | Robust regression | `nn.SmoothL1Loss()` |
| **CTC** | Sequence | Speech/OCR | Alignment-free sequences | `nn.CTCLoss()` |
| **Perceptual** | Image | Style/Quality | Perceptual image tasks | Custom (VGG) |
| **Triplet** | Metric | Face recognition | Learn embeddings | `nn.TripletMarginLoss()` |
| **IoU** | Detection | Bounding boxes | Object detection | Custom |
| **Hinge** | GAN/SVM | Image synthesis | High-quality GANs | Custom |

---

## Practical Tips & Common Pitfalls

### ⚡ General Best Practices

1. **Start Simple**: Begin with MSE/Cross Entropy before exotic losses
2. **Combine Losses**: Often `loss = α·loss1 + β·loss2` works better
3. **Scale Matters**: Normalize inputs, outputs, and loss magnitudes
4. **Monitor Components**: Log each loss term separately
5. **Validate on Metric**: Loss ≠ performance (track accuracy, mAP, etc.)

### ⚠️ Common Mistakes

**Cross Entropy:**
- ❌ Applying softmax before `CrossEntropyLoss` (it's included!)
- ❌ Using one-hot targets (use class indices)
- ✅ Use `label_smoothing=0.1` for regularization

**BCE:**
- ❌ Using `BCELoss` with logits (numerical instability)
- ✅ Always use `BCEWithLogitsLoss`

**Focal Loss:**
- ❌ Using on balanced datasets (unnecessary)
- ❌ γ too high (training instability)
- ✅ Start with α=0.25, γ=2.0

**Contrastive/InfoNCE:**
- ❌ Too small batch size (<32)
- ❌ No normalization of embeddings
- ✅ Use large batches, tune temperature carefully

**Dice Loss:**
- ❌ Using alone (poor gradients at boundaries)
- ✅ Combine with Cross Entropy (0.5 each)

**Wasserstein:**
- ❌ Using batch norm in critic (breaks Lipschitz)
- ❌ Weight clipping without gradient penalty
- ✅ Use WGAN-GP or Spectral Normalization

### 🎯 Loss Selection Flowchart

```
Task Type?
├─ Regression
│  ├─ Clean data, no outliers → MSE
│  ├─ Outliers present → Huber/Smooth L1
│  └─ Bounding boxes → IoU Loss
│
├─ Classification
│  ├─ Binary → BCEWithLogitsLoss
│  ├─ Multi-class balanced → CrossEntropyLoss
│  ├─ Multi-class imbalanced → Focal Loss
│  └─ Multi-label → BCEWithLogitsLoss (per label)
│
├─ Segmentation
│  ├─ Balanced → CrossEntropyLoss
│  ├─ Imbalanced/small objects → Dice + CE
│  └─ Bounding boxes → IoU variants
│
├─ Sequence
│  ├─ Fixed alignment → CrossEntropyLoss
│  └─ Unknown alignment → CTC Loss
│
├─ Embedding/Similarity
│  ├─ Pairs → Contrastive Loss
│  ├─ Triplets → Triplet Loss
│  └─ Multi-modal → InfoNCE (CLIP-style)
│
└─ Generative
   ├─ VAE → Reconstruction + KL
   ├─ GAN → Wasserstein (WGAN-GP) or Hinge
   ├─ Diffusion → MSE (denoising)
   └─ Style/Quality → Perceptual Loss
```

### 🔬 Advanced Techniques

**Loss Weighting Strategies:**
- Manual: `loss = 0.5·L1 + 0.5·L2`
- Uncertainty weighting: Learn weights automatically
- Dynamic weighting: Change during training (annealing)

**Hard Example Mining:**
- Focal Loss: Automatic hard example focus
- OHEM: Online Hard Example Mining
- Hard negative mining: For Triplet Loss

**Curriculum Learning:**
- Start with easy examples, gradually harder
- Loss annealing: Gradually increase difficult term

---

## Summary: What to Remember

### 🔴 Top Priority (Master These)
1. **MSE** - Your regression default
2. **Cross Entropy** - Your classification default
3. **BCE** - Binary classification standard
4. **Focal Loss** - For any imbalanced problem
5. **Dice Loss** - For any segmentation
6. **InfoNCE / Contrastive** - For modern embeddings (CLIP, SimCLR)
7. **KL Divergence** - For VAEs and distillation
8. **Wasserstein** - For stable GAN training

### 🟡 High Priority (Know Well)
9. **Smooth L1 / Huber** - For robust regression & detection
10. **CTC Loss** - For speech recognition & OCR
11. **Perceptual Loss** - For image quality tasks
12. **Triplet Loss** - For face recognition & embeddings
13. **IoU variants** - For bounding box regression
14. **Hinge Loss** - For modern GANs (StyleGAN)

