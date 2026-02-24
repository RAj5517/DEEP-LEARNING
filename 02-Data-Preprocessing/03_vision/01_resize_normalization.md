# Resize + Normalization

The first two steps convert raw images — which arrive in wildly different sizes, formats, and pixel ranges — into a consistent, numerically stable form the model can process.

---

## 1. Resize

### Why
CNNs require fixed-size input. Every image in a batch must have identical dimensions. Raw images from cameras, web scrapes, or datasets vary from 64×64 to 4000×3000.

### Standard Sizes by Architecture

| Model | Input Size |
|---|---|
| AlexNet | 227×227 |
| VGG16 / VGG19 | 224×224 |
| ResNet (all variants) | 224×224 |
| EfficientNet-B0 | 224×224 |
| EfficientNet-B7 | 600×600 |
| Vision Transformer (ViT-B) | 224×224 |
| InceptionV3 | 299×299 |
| YOLO v8 | 640×640 |

### Interpolation Methods

The algorithm used to resize matters — it affects edge sharpness and training quality.

| Method | Quality | Speed | Use When |
|---|---|---|---|
| `BILINEAR` | Good | Fast | Default for training |
| `BICUBIC` | Better | Slower | ViT, high-quality inference |
| `NEAREST` | Poor | Fastest | Segmentation masks (preserves label IDs) |
| `LANCZOS` | Best | Slowest | Offline preprocessing |

### Implementation

```python
from torchvision import transforms
from PIL import Image

# Standard resize for CNN
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # bilinear by default
    transforms.ToTensor()
])

# Resize then center crop (preserves aspect ratio)
transform = transforms.Compose([
    transforms.Resize(256),          # resize short edge to 256
    transforms.CenterCrop(224),      # crop center 224×224
])
```

### Resize vs CenterCrop
- **Resize directly** → stretches/squashes the image, changes aspect ratio
- **Resize then CenterCrop** → preserves aspect ratio, discards borders — preferred for inference
- **RandomResizedCrop** → resizes + crops randomly — preferred for training augmentation

---

## 2. Normalization

### Why
Raw pixels range from `[0, 255]`. Neural networks with gradient descent converge faster and more stably when inputs are small, zero-centered values. Without normalization:
- Gradients in early layers are dominated by large pixel values
- Weight updates become unstable
- Convergence is slow or diverges

### Step 1 — Scale to [0, 1]

Divide by 255 to bring pixels from integer range into float range.

```python
# torchvision.transforms.ToTensor() does this automatically
# It converts HWC uint8 [0,255] → CHW float [0,1]
tensor = transforms.ToTensor()(image)
```

### Step 2 — Mean/Std Normalization

Standardize each channel using dataset-level mean and standard deviation.

```
x' = (x - mean) / std
```

This centers the data around 0 and scales to approximately [-1, 1].

```python
# ImageNet statistics — use these when fine-tuning pretrained models
mean = [0.485, 0.456, 0.406]   # R, G, B channel means
std  = [0.229, 0.224, 0.225]   # R, G, B channel stds

normalize = transforms.Normalize(mean=mean, std=std)
```

### Computing Your Own Dataset Statistics

If training from scratch on a custom dataset, compute stats from training data:

```python
import torch
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=False)
mean = torch.zeros(3)
std  = torch.zeros(3)

for images, _ in loader:
    # images: (B, C, H, W)
    mean += images.mean(dim=[0, 2, 3])
    std  += images.std(dim=[0, 2, 3])

mean /= len(loader)
std  /= len(loader)
print(f"Mean: {mean}")
print(f"Std:  {std}")
```

### Common Normalization Values

| Dataset / Use case | Mean | Std |
|---|---|---|
| ImageNet (pretrained) | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |
| CIFAR-10 | [0.491, 0.482, 0.447] | [0.247, 0.243, 0.261] |
| MNIST (grayscale) | [0.1307] | [0.3081] |
| Simple [0,1] scaling | [0.5, 0.5, 0.5] | [0.5, 0.5, 0.5] |

### Full Transform Pipeline

```python
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),                         # → [0, 1]
    transforms.Normalize(                          # → ~[-1, 1]
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])
```