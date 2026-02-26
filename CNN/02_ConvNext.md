# ConvNeXt — A ConvNet for the 2020s (2022)

Liu et al., Meta AI — "We gradually 'modernize' a standard ResNet toward the design of a Vision Transformer."

---

## The Problem It Solved

By 2021, Vision Transformers (ViT, Swin) were outperforming CNNs on most benchmarks. The prevailing narrative: **attention mechanisms are fundamentally better than convolutions for vision**.

ConvNeXt challenged this by asking a different question: **what if the CNN's underperformance wasn't about convolutions, but about outdated design choices?**

The paper took ResNet-50 and systematically applied every modern design decision learned from the Transformer era — one change at a time — to see which changes drove the improvement. The result matched or outperformed Swin Transformer at every model size using only standard convolutions, no attention required.

---

## The Modernization Roadmap

Starting from ResNet-50 (78.8% ImageNet accuracy), each change was applied sequentially:

```
ResNet-50 baseline                          78.8%
  │
  ├── 1. Training recipe (AdamW, cosine, mixup, etc.)     → 79.9% (+1.1)
  ├── 2. Macro design: stage ratio 1:1:3:1 → 1:1:9:1     → 80.6% (+0.7)  
  ├── 3. ResNeXt: grouped convolutions (depthwise)        → 80.5%
  ├── 4. Inverted bottleneck (wide → narrow → wide)       → 80.5%
  ├── 5. Large kernel: 3×3 → 7×7 depthwise               → 80.6%
  ├── 6. Micro design: ReLU → GELU, fewer activations     → 81.3% (+0.7)
  └── 7. BatchNorm → LayerNorm                            → 81.3%

ConvNeXt-T final                            82.1%
Swin-T (comparison)                         81.3%
```

Each change came from analyzing why Transformers perform better and finding the CNN equivalent.

---

## Key Design Changes (ResNet → ConvNeXt)

### 1. Stage Ratio Change
ResNet distributes blocks as (3, 4, 6, 3). Swin uses (1, 1, 9, 1) — most compute in stage 3. ConvNeXt adopts (3, 3, 9, 3).

### 2. Depthwise Convolution (from ResNeXt)
Replace standard 3×3 conv with **depthwise** 7×7 conv — one filter per channel, larger kernel, far fewer parameters. This is the CNN equivalent of self-attention's "mixing information across the spatial dimension independently per channel."

### 3. Inverted Bottleneck
Transformers' FFN block expands dimension 4× then contracts. ConvNeXt flips the ResNet bottleneck: expand channels wide in the middle, contract at the ends.

```
ResNet Bottleneck:    256 → 64 → 64 → 256  (narrow in middle)
ConvNeXt Block:        96 → 384 → 96        (wide in middle, like Transformer FFN)
```

### 4. Large Kernel (7×7)
Moving from 3×3 to 7×7 depthwise conv increases receptive field with minimal cost (because depthwise is cheap). Matches the effect of self-attention's global receptive field at smaller scales.

### 5. GELU Instead of ReLU
Transformers use GELU everywhere. Swapping ReLU → GELU provides a small consistent improvement.

### 6. LayerNorm Instead of BatchNorm
BatchNorm depends on batch statistics, behaves differently at train vs test time, and struggles with small batches. LayerNorm normalizes per sample — simpler, more consistent, works at any batch size. Transformers use LayerNorm exclusively. ConvNeXt adopts it, applied before the conv (pre-norm, like modern Transformers).

---

## The ConvNeXt Block

```
Input x
  │
  ▼
DWConv2d(C, C, 7×7, padding=3)    ← Large kernel depthwise conv (spatial mixing)
LayerNorm(C)                        ← Normalize per sample
Linear(C, 4C)                       ← Expand (pointwise, 1×1 equivalent)
GELU()
Linear(4C, C)                       ← Contract (pointwise)
  │
  ├── Scale by learnable γ (LayerScale — stabilizes training)
  │
(+) ← residual from input x
  │
  ▼
Output x
```

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale=1e-6):
        super().__init__()
        self.dwconv   = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm     = nn.LayerNorm(dim)
        self.pwconv1  = nn.Linear(dim, 4 * dim)   # Pointwise expand
        self.act      = nn.GELU()
        self.pwconv2  = nn.Linear(4 * dim, dim)   # Pointwise contract
        self.gamma    = nn.Parameter(layer_scale * torch.ones(dim))
    
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)     # [B,C,H,W] → [B,H,W,C] for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)     # [B,H,W,C] → [B,C,H,W]
        return x + residual
```

---

## ConvNeXt Architecture

```
Input [3 × 224 × 224]
       │
       ▼
Stem: Conv2d(3, C, 4×4, stride=4)     → [C × 56 × 56]  (patchify like ViT)
LayerNorm
       │
       ▼
Stage 1: C channels,    3 blocks       → [C × 56 × 56]
Downsample: LN + Conv2d(C, 2C, 2×2, stride=2)
Stage 2: 2C channels,   3 blocks       → [2C × 28 × 28]
Downsample
Stage 3: 4C channels,   9 blocks       → [4C × 14 × 14]
Downsample
Stage 4: 8C channels,   3 blocks       → [8C × 7 × 7]
       │
       ▼
Global Average Pooling → LayerNorm → Linear(8C, num_classes)
```

**Model variants (C = base channel width):**

| Model | C | Blocks | Params | ImageNet Top-1 |
|-------|---|--------|--------|----------------|
| ConvNeXt-T (Tiny) | 96 | 3,3,9,3 | 29M | 82.1% |
| ConvNeXt-S (Small) | 96 | 3,3,27,3 | 50M | 83.1% |
| ConvNeXt-B (Base) | 128 | 3,3,27,3 | 89M | 83.8% |
| ConvNeXt-L (Large) | 192 | 3,3,27,3 | 198M | 84.3% |
| ConvNeXt-XL | 256 | 3,3,27,3 | 350M | 84.6% |

---

## PyTorch Usage

```python
import torchvision.models as models

# ConvNeXt-T (good default)
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

# Fine-tune for custom task
num_classes = 10
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

# ConvNeXt works well with AdamW (not SGD)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=5e-5, weight_decay=0.05
)
```

**Note:** ConvNeXt trains better with AdamW than SGD — unlike traditional CNNs. This is one of the Transformer-era design borrowings that carries over.

---

## ConvNeXt vs Swin Transformer

| | ConvNeXt | Swin Transformer |
|--|----------|-----------------|
| Mechanism | Depthwise Conv | Shifted Window Attention |
| Accuracy (same FLOPs) | Comparable | Comparable |
| Implementation | Simple | More complex |
| Inference speed | Faster | Slower |
| Arbitrary resolution | Natural | Requires adaptation |
| Dense prediction (detection/seg) | Good | Good |

For most practical applications: **ConvNeXt is simpler to use and equally performant**. The theoretical elegance of attention doesn't translate to a meaningful accuracy gap at standard scales.

---

## Key Takeaway

ConvNeXt's real contribution isn't a new architecture — it's a proof that **convolutions aren't the bottleneck**. The gap between CNNs and Transformers was mostly about training recipes and design choices, not the fundamental operation. If you need a strong vision backbone today and want simplicity over the latest trend, ConvNeXt is the answer.