# CNN — Convolutional Neural Network

---

## What Is It?

A Convolutional Neural Network is a neural network designed specifically for **data with spatial structure** — primarily images, but also audio spectrograms, video, and any 2D/3D grid-like data.

The key idea: instead of connecting every neuron to every input pixel (which would be enormous — a 224×224 image has 150,528 values), CNNs use **local filters** that slide across the input, learning to detect patterns regardless of where they appear in the image. This gives CNNs two fundamental properties:

**Local connectivity** — each filter only looks at a small region (e.g., 3×3 pixels) at a time, learning local patterns like edges, textures, curves.

**Weight sharing** — the same filter is applied at every position in the image. An edge detector learned in one corner applies everywhere. This massively reduces parameters compared to a fully connected network.

The result: CNNs build a **hierarchy of features** — early layers detect simple patterns (edges, colors), middle layers combine them into textures and shapes, deep layers recognize complex objects. This hierarchy mirrors how the visual cortex works.

---

## Architecture

### The Core Operations

#### Convolution

A filter (kernel) slides across the input, computing a dot product at each position:

```
Input (5×5):          Filter (3×3):       Output (3×3):
┌─────────────┐       ┌───────────┐       ┌───────────┐
│ 1  2  3  0  1│      │ 1  0  -1 │       │    6      │
│ 0  1  2  3  0│  *   │ 1  0  -1 │   =   │   ...     │
│ 1  0  1  2  3│      │ 1  0  -1 │       │   ...     │
│ 0  1  0  1  2│      └───────────┘       └───────────┘
│ 1  2  1  0  1│
└─────────────┘
```

Each filter detects one specific pattern. A layer with 64 filters produces 64 different **feature maps** — 64 different pattern detectors applied to the same input.

**Key parameters:**
- `kernel_size`: size of the filter (3×3, 5×5, 7×7)
- `stride`: how many pixels to move the filter each step (stride=1: dense, stride=2: halves spatial size)
- `padding`: add zeros around border to control output size

```
Output size = (Input - Kernel + 2·Padding) / Stride + 1
```

#### Pooling

Downsamples spatial dimensions, reducing computation and providing local translation invariance:

```
Max Pooling (2×2, stride=2):         Average Pooling:
┌───────────┐                        ┌───────────┐
│ 1  3  2  4│     →   ┌─────┐        │ 1  3  2  4│    →   ┌─────┐
│ 5  6  1  2│         │ 6  4│        │ 5  6  1  2│        │3.75 2.25│
│ 3  2  4  1│         │ 4  4│        │ 3  2  4  1│        │2.75 2.5 │
│ 1  2  3  4│         └─────┘        │ 1  2  3  4│        └─────┘
└───────────┘                        └───────────┘
Takes maximum in each region         Takes average in each region
```

MaxPooling is most common in CNNs. In modern architectures it's often replaced by strided convolutions.

---

### Full CNN Flow

```
Input Image
[3 × 224 × 224]
       │
       ▼
┌─────────────────┐
│  Conv Layer 1   │  64 filters, 3×3, ReLU   →  [64 × 224 × 224]
│  BatchNorm      │
│  ReLU           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Conv Layer 2   │  128 filters, 3×3, ReLU  →  [128 × 112 × 112]
│  BatchNorm      │  (stride=2 halves spatial size)
│  ReLU           │
└────────┬────────┘
         │
         ▼
    [... more conv layers, spatial size shrinks, channels grow ...]
         │
         ▼
┌─────────────────┐
│ Global Avg Pool │  →  [512 × 1 × 1]  →  flatten  →  [512]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FC / MLP Head  │  →  [num_classes]
└─────────────────┘
```

**The pattern:** Spatial dimensions shrink (224→112→56→28→14→7), channel dimensions grow (3→64→128→256→512). This is the CNN compression: translate spatial information into feature-channel information.

---

## Key Concepts

### Receptive Field

The region of the original input image that influences a particular neuron. Deep in the network, neurons have large receptive fields and respond to global patterns.

```
Layer 1 (3×3 conv):  sees 3×3 pixels of input
Layer 2 (3×3 conv):  sees 5×5 pixels of input (because layer 1 already aggregated)
Layer 3 (3×3 conv):  sees 7×7 pixels of input
```

Stacking small (3×3) filters builds large receptive fields efficiently with fewer parameters than one large filter.

### Feature Maps

The output of each convolutional layer is a set of feature maps — one per filter. Each feature map is a spatial activation grid showing **where** in the image that filter's pattern was detected.

```
Input: photo of a cat
After conv layer: 64 feature maps
  - Map 1: activates strongly at horizontal edges
  - Map 5: activates at fur texture regions
  - Map 23: activates at round shapes (eyes, ears)
  ...
```

### 1×1 Convolution

A convolution with a 1×1 kernel — operates only on channels, not spatial dimensions. Used to:
- Change the number of channels cheaply (dimensionality reduction)
- Add nonlinearity without affecting spatial structure
- The key operation in Inception modules and bottleneck blocks

### Depthwise Separable Convolution

Split standard convolution into two steps:
1. **Depthwise**: apply one filter per channel independently (spatial mixing)
2. **Pointwise**: 1×1 conv to mix channels

Dramatically reduces parameters (~8-9× cheaper than standard conv). Used in MobileNet and EfficientNet for efficient architectures.

```python
# Standard conv: in_channels × out_channels × K × K parameters
# Depthwise separable:
nn.Conv2d(C, C, kernel_size=3, groups=C)  # Depthwise (one filter per channel)
nn.Conv2d(C, C_out, kernel_size=1)         # Pointwise (channel mixing)
```

---

## PyTorch Implementation

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # 224 → 112
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # 112 → 56
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # 56 → 28
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # Global average pooling → [256, 1, 1]
            nn.Flatten(),                   # → [256]
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

## CNN Types — Family Tree

```
CNN Family
│
├── Classic / Foundational
│     ├── LeNet (1998)         → Proof of concept, MNIST
│     ├── AlexNet (2012)       → Deep learning revolution, ImageNet
│     ├── VGG (2014)           → Depth with simple 3×3 filters
│     └── GoogLeNet (2014)     → Multi-scale (Inception module)
│
├── Residual Era           ← The turning point
│     ├── ResNet (2015) ⭐  → Skip connections, scalable depth
│     └── DenseNet (2017)   → Every layer connects to every other
│
├── Efficiency Era
│     ├── MobileNet (2017)     → Depthwise separable, edge devices
│     └── EfficientNet (2019)  → Compound scaling (depth+width+resolution)
│
└── Modern Era             ← Current standard
      └── ConvNeXt (2022) ⭐ → CNN redesigned to match Transformers
```

**Deep dives:**
- ResNet → `resnet.md`
- ConvNeXt → `convnext.md`
- All others (LeNet through EfficientNet) → `cnn_variants.md`

---

## When to Use CNN

| Situation | Recommendation |
|-----------|---------------|
| Image classification | ResNet-50 or ConvNeXt-B as backbone |
| Object detection | ResNet/ConvNeXt backbone + FPN + detection head |
| Segmentation | U-Net (CNN encoder-decoder) |
| Edge / mobile | MobileNet or EfficientNet-B0 |
| Transfer learning | Pretrained ResNet or ConvNeXt from torchvision |
| From scratch, small data | Simple CNN or ResNet-18 |

CNNs remain competitive with Vision Transformers (ViT) at medium scale and smaller datasets, and are more efficient for many practical applications.