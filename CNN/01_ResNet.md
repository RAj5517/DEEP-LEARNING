# ResNet — Deep Residual Networks (2015)

He et al., Microsoft Research — Won ILSVRC 2015 with 152 layers and 3.57% top-5 error.

---

## The Problem It Solved

Before ResNet, adding more layers made networks **worse** — not just slower to train, but actually less accurate even on training data. This was not a generalization problem. It was an optimization problem.

The hypothesis: very deep networks are hard for gradient-based optimization to train. Gradients must travel through dozens of multiplications during backpropagation, shrinking exponentially by the time they reach early layers. Early layers stop learning.

**ResNet's solution:** Give gradients a direct path to flow backward through the network — a "shortcut" that bypasses layers entirely.

---

## The Residual Connection

Instead of learning `H(x)` (the full desired mapping), let the layer learn the **residual** `F(x) = H(x) - x`. The output becomes:

```
output = F(x) + x
```

```
Without residual:           With residual:
                            
x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → y
                                                    
x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → (+) → y
└──────────────────────────────────────────────┘
                      identity shortcut
```

**Why this works:**
- Gradients flow directly through the `+ x` path without any multiplication — no vanishing
- If the layers aren't needed (identity is optimal), they can easily learn `F(x) = 0`
- The network never gets worse by adding layers — at worst, extra layers learn identity

---

## Building Block: The Residual Block

### Basic Block (ResNet-18, ResNet-34)

```
x ─────────────────────────────────────┐
│                                       │
▼                                       │
Conv2d(C, C, 3×3, pad=1)               │
BatchNorm2d                             │
ReLU                                    │
Conv2d(C, C, 3×3, pad=1)               │
BatchNorm2d                             │
                                        │ (identity shortcut)
(+) ← ─────────────────────────────────┘
▼
ReLU
```

```python
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual    # Skip connection
        return self.relu(out)
```

### Bottleneck Block (ResNet-50, ResNet-101, ResNet-152)

For deeper networks, a 3-layer bottleneck reduces computation:

```
x ─────────────────────────────────────────────────────┐
│                                                        │
Conv2d(C, C/4, 1×1)  ← reduce channels (bottleneck)    │
BatchNorm, ReLU                                         │
Conv2d(C/4, C/4, 3×3) ← spatial convolution            │  1×1 conv to match dims
BatchNorm, ReLU                                         │  (if channels change)
Conv2d(C/4, C, 1×1)  ← restore channels                │
BatchNorm                                               │
(+) ← ──────────────────────────────────────────────────┘
ReLU
```

The bottleneck uses a 1×1 → 3×3 → 1×1 pattern. The 1×1 convolutions handle channel changes cheaply; the expensive 3×3 operates on reduced channels. Nearly 4× fewer FLOPs than the basic block for the same output.

**When channels or spatial size changes** (between stages), the shortcut uses a 1×1 conv to match dimensions:

```python
self.shortcut = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
    nn.BatchNorm2d(out_channels)
)
```

---

## ResNet Architecture

```
Input [3 × 224 × 224]
       │
       ▼
Conv2d(3, 64, 7×7, stride=2, pad=3)  → [64 × 112 × 112]
BatchNorm, ReLU
MaxPool(3×3, stride=2)               → [64 × 56 × 56]
       │
       ▼
Stage 1: 64 channels,  N blocks      → [64 × 56 × 56]
Stage 2: 128 channels, N blocks      → [128 × 28 × 28]  (stride=2 at start)
Stage 3: 256 channels, N blocks      → [256 × 14 × 14]  (stride=2 at start)
Stage 4: 512 channels, N blocks      → [512 × 7 × 7]    (stride=2 at start)
       │
       ▼
Global Average Pooling               → [512]
Linear(512, num_classes)             → [num_classes]
```

**N blocks per stage by variant:**

| Model | Stage blocks | Total layers | Params | Use case |
|-------|-------------|-------------|--------|----------|
| ResNet-18 | 2,2,2,2 | 18 | 11M | Fast, small tasks, mobile |
| ResNet-34 | 3,4,6,3 | 34 | 21M | Moderate tasks |
| ResNet-50 | 3,4,6,3 | 50 | 25M | **Default ⭐** |
| ResNet-101 | 3,4,23,3 | 101 | 44M | Higher accuracy |
| ResNet-152 | 3,8,36,3 | 152 | 60M | Maximum accuracy |

ResNet-50 is the sweet spot — strong performance, reasonable size, widely benchmarked.

---

## PyTorch Usage

```python
import torchvision.models as models

# Load pretrained ResNet-50 (ImageNet weights)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Fine-tune: replace final layer for your task
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze backbone (optional — for small datasets)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)   # Only train the head

# Full fine-tuning (for larger datasets)
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                             momentum=0.9, weight_decay=1e-4)
```

---

## Key Contributions and Legacy

**What ResNet proved:**
- Networks can be made arbitrarily deep if you solve the gradient flow problem
- Residual learning is the right inductive bias for deep networks — it's now in everything
- Simple architecture decisions (BatchNorm + skip connections) beat complex hand-engineering

**Where residual connections appear today:**
- Every Transformer block: `x = x + Attention(x)` and `x = x + FFN(x)` — both are residual
- U-Net skip connections (a form of cross-layer residuals)
- DenseNet (dense residuals)
- ConvNeXt, EfficientNet — all use residuals

ResNet didn't just win a competition — it changed the design philosophy of deep learning architectures permanently.