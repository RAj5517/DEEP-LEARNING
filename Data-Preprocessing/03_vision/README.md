# 🖼️ Computer Vision — Image Preprocessing

> Raw images are noisy, inconsistently sized, and pixel-heavy.
> Models need uniform, numerically stable tensors.
> Everything between a raw image file and model input is **CV preprocessing**.

---

## 🔁 Pipeline at a Glance

```
Raw Images  (different sizes, lighting, noise)
   ↓
Resize               (uniform spatial dimensions)
   ↓
Normalization        (stable pixel value range)
   ↓
Data Augmentation    (training only — improves generalization)
   ↓
Tensor Conversion    (HWC uint8 → CHW float32 → GPU)
   ↓
CNN / Vision Transformer
```

---

## 📂 Structure

| File | Covers |
|---|---|
| `01_resize_normalization.md` | Resize strategies, interpolation, pixel scaling, mean/std |
| `02_data_augmentation.md` | Geometric, color, noise, Mixup, CutMix, AutoAugment |
| `03_tensor_conversion.md` | HWC→CHW, PIL/OpenCV/NumPy, GPU transfer, dtypes |

---

## ❓ Why This Matters

| Problem | Caused By |
|---|---|
| Runtime crash (size mismatch) | Variable image sizes — no resize |
| Slow / unstable convergence | Raw [0,255] pixels — no normalization |
| Overfitting | No augmentation — model memorizes training images |
| Wrong colors (bad predictions) | OpenCV BGR not converted to RGB |
| Model doesn't use GPU | Tensor not moved to device |

---

## ⚡ When to Apply Each Step

| Step | Training | Validation | Inference |
|---|---|---|---|
| Resize | ✅ | ✅ | ✅ |
| Normalization | ✅ | ✅ | ✅ |
| Data Augmentation | ✅ | ❌ | ❌ |
| Tensor Conversion | ✅ | ✅ | ✅ |

---

## 🔬 Core Idea

Every image goes through one transformation:
**pixel grid → normalized tensor → model-ready batch.**

Augmentation is the only step that varies between train and eval.
Everything else is deterministic and applied identically every time.

---

*For deep breakdowns, math, and code — refer to the individual files above.*

┌──────────────────────────────────────────────┐
  │     RAW IMAGES  (different sizes, lighting)  │
  └──────────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │              1. RESIZE                       │
  └──────────────────────────────────────────────┘
  │
  ├── Fixed size input required by all CNNs
  │
  ├── Common sizes
  │     ├── 224×224   ResNet · VGG · ViT · EfficientNet-B0
  │     ├── 299×299   InceptionV3
  │     └── 640×640   YOLO
  │
  ├── Resize directly         → stretches image
  └── Resize → CenterCrop    → preserves aspect ratio  ← preferred
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │           2. NORMALIZATION                   │
  └──────────────────────────────────────────────┘
  │
  ├── Step 1 — Scale pixels
  │     pixel / 255  →  [0, 1]      (ToTensor does this)
  │
  └── Step 2 — Mean / Std Normalization
        x' = (x − mean) / std       →  ~[−1, 1]
        ImageNet:  mean=[0.485, 0.456, 0.406]
                   std =[0.229, 0.224, 0.225]
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │         3. DATA AUGMENTATION                 │
  └──────────────────────────────────────────────┘
  │                              training only ↑
  ├── Geometric
  │     ├── Horizontal Flip
  │     ├── Random Crop  (RandomResizedCrop)
  │     ├── Rotation
  │     └── Affine / Perspective
  │
  ├── Color / Photometric
  │     ├── Color Jitter  (brightness, contrast, saturation, hue)
  │     ├── Grayscale
  │     └── Gaussian Blur
  │
  ├── Noise / Masking
  │     ├── Gaussian Noise
  │     └── Cutout         (mask random square region)
  │
  ├── Sample Mixing
  │     ├── Mixup          (blend two images + labels)
  │     └── CutMix         (paste region from another image)
  │
  └── Auto Policies
        ├── AutoAugment    (learned policy)
        ├── RandAugment    (random N ops of magnitude M)
        └── TrivialAugment (one random op per image)
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │          4. TENSOR CONVERSION                │
  └──────────────────────────────────────────────┘
  │
  ├── PIL / NumPy  (H, W, C)  uint8  [0, 255]
  │         ↓   transforms.ToTensor()
  ├── Tensor       (C, H, W)  float32  [0.0, 1.0]
  │
  ├── Add batch dim   →  unsqueeze(0)  →  (1, C, H, W)
  ├── Move to GPU     →  tensor.to(device)
  └── dtype           →  float32  (default)  ·  float16 (AMP)
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │         CNN / VISION TRANSFORMER             │
  └──────────────────────────────────────────────┘