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
