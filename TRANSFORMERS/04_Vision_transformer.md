# Vision Transformer (ViT)

> "What if we stopped treating images as pixels — and treated them as words?"

---

## The Core Idea

CNNs dominated computer vision for a decade. They slide a small filter across an image, detecting edges → shapes → objects layer by layer.

In 2020, Google Brain asked: what if we just used a Transformer on images directly?

The problem: Transformers work on sequences of tokens. Images are 2D grids of pixels. A 224×224 image has 50,176 pixels — way too many to attend to one by one.

**The solution: Split the image into patches. Treat each patch as a token.**

A 224×224 image split into 16×16 patches = **196 tokens**. Feed those 196 tokens into a standard Transformer encoder. Done.

---

## Architecture

```
Input Image (e.g. 224×224×3)
│
├── Split into patches             16×16 patches → 196 patches
├── Flatten each patch             16×16×3 = 768 values per patch
├── Linear Projection              768 → embedding dim (e.g. 768)
│
├── Add [CLS] token                prepended — will carry classification info
├── Add Positional Embedding       1D learnable position per patch
│
└── × N Transformer Encoder Layers
      ├── Multi-Head Self-Attention    patches attend to each other
      ├── Add & LayerNorm
      ├── Feed Forward Network (MLP)
      └── Add & LayerNorm
│
└── [CLS] token output
└── MLP Classification Head → class label
```

---

## How Patches Become Tokens

```
Original image
┌─────────────────┐
│  🐱             │
│                 │   224×224 pixels
│                 │
└─────────────────┘
         ↓  split into 16×16 patches
┌──┬──┬──┬──┬──┬──┐
│P1│P2│P3│P4│P5│P6│   each patch = one "word"
├──┼──┼──┼──┼──┼──┤
│P7│P8│...         │   196 patches total
└──┴──┴──┴──┴──┴──┘
         ↓  flatten + linear projection
[vec1, vec2, vec3, ... vec196]   ← sequence of tokens
         ↓  standard transformer encoder
[CLS] vector → "Persian Cat, 94%"
```

---

## Self-Attention in Vision

Because every patch attends to every other patch, ViT naturally learns **long-range dependencies** — something CNNs struggle with since they only have local receptive fields.

Patch at top-left of image can directly attend to patch at bottom-right in layer 1.
CNNs would need many convolutional layers to connect such distant regions.

This gives ViT an advantage on tasks requiring **global understanding** of the image.

---

## ViT vs CNN — Key Differences

```
CNN                              Vision Transformer
│                                │
├── Local filters (3×3, 5×5)    ├── Global attention (all patches)
├── Translation equivariant      ├── No inductive bias — learns from data
├── Works with small data        ├── Needs large data (or pretraining)
├── Efficient on CPU/GPU         ├── Needs GPU/TPU for large models
└── Strong inductive bias        └── More flexible, better at scale
```

**Inductive bias**: CNNs "know" that nearby pixels are related. ViT has no such assumption — it learns spatial relationships from scratch. This means ViT needs more data, but is more flexible.

---

## Use Cases

```
Vision Transformer Use Cases
│
├── Image Classification         ImageNet · medical imaging · satellite
├── Object Detection             DETR · Swin-based detectors
├── Semantic Segmentation        SegFormer · Mask2Former
├── Image Generation             DiT (Diffusion Transformer) — Stable Diffusion 3
├── Video Understanding          VideoMAE · TimeSformer
└── Medical Imaging              PathologyViT · RadViT · cell classification
```

---

## Main Models

### ViT — Vision Transformer (2020) — Google Brain
The original. Showed that a pure Transformer with no convolutions can match or beat CNNs on ImageNet — but only when pre-trained on very large datasets (JFT-300M, 300 million images).

- Patch size: 16×16 or 32×32
- Sizes: ViT-Base (86M) · ViT-Large (307M) · ViT-Huge (632M)
- Paper: https://arxiv.org/abs/2010.11929

---

### DeiT — Data-efficient Image Transformers (2020) — Facebook AI
ViT needed 300M images to train well. DeiT showed you could train on ImageNet alone (1.2M images) using **knowledge distillation** from a CNN teacher model.

Made ViT practical for researchers without massive compute budgets.
Paper: https://arxiv.org/abs/2012.12877

---

### Swin Transformer (2021) — Microsoft
**Shifted Window Transformer** — the most practically impactful ViT variant.

Key innovations:
- **Hierarchical features** — like CNNs, builds from small patches to larger representations
- **Local windowed attention** — instead of global attention, each patch attends only to its local window (7×7 patches)
- **Shifted windows** — windows shift between layers to allow cross-window communication

Result: Much more efficient than full ViT. Became backbone of choice for detection and segmentation.
Paper: https://arxiv.org/abs/2103.14030

---

### BEiT — BERT Pre-Training for Image Transformers (2021) — Microsoft
Applied BERT's masked language modeling to images. Tokenize image into visual tokens (using DALL-E's discrete VAE), mask random patches, make model predict the visual tokens. Strong self-supervised pre-training for vision.

Paper: https://arxiv.org/abs/2106.08254

---

### MAE — Masked Autoencoders (2021) — Meta / FAIR
Even simpler: mask 75% of image patches and make the model reconstruct the original pixels. Very compute-efficient pre-training. Used as foundation for many downstream vision tasks.

Paper: https://arxiv.org/abs/2111.06377

---

### DiT — Diffusion Transformer (2022)
Replaced the U-Net backbone in diffusion models with a Transformer. Used as architecture in Stable Diffusion 3, SORA (video), and modern image generation pipelines.

Paper: https://arxiv.org/abs/2212.09748

---

## Key Papers

| Paper | Link |
|-------|------|
| ViT (2020) | https://arxiv.org/abs/2010.11929 |
| DeiT (2020) | https://arxiv.org/abs/2012.12877 |
| Swin Transformer (2021) | https://arxiv.org/abs/2103.14030 |
| BEiT (2021) | https://arxiv.org/abs/2106.08254 |
| MAE (2021) | https://arxiv.org/abs/2111.06377 |
| DiT (2022) | https://arxiv.org/abs/2212.09748 |