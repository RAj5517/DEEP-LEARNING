# CNN Variants — LeNet to EfficientNet

Short reference for the CNN architectures that shaped the field. Understand the problem each solved and what it contributed — the details live in ResNet and ConvNeXt files.

---

## LeNet (1998) — LeCun et al.

**Problem solved:** Proved CNNs can learn to recognize handwritten digits automatically.

**Architecture:** 2 conv layers + 2 pooling layers + 2 FC layers. Input: 32×32 grayscale. Used tanh/sigmoid activations, average pooling.

**Why it matters:** The conceptual blueprint every CNN follows — conv → pool → conv → pool → flatten → classify. Everything after is a scaled-up, improved version of this.

**Used today:** Not in practice. Historical significance only.

---

## AlexNet (2012) — Krizhevsky, Sutskever, Hinton

**Problem solved:** Proved deep CNNs can beat traditional computer vision on large-scale datasets. Won ImageNet (ILSVRC 2012) by ~10% over second place — a shock to the field.

**Key contributions over LeNet:**
- First use of **ReLU** (replaced sigmoid/tanh) — faster training, no vanishing gradient for moderate depth
- **Dropout** (0.5) in FC layers
- **Data augmentation** (random crops, flips, color jitter)
- **GPU training** — used two GTX 580 GPUs, split the model across them
- **Local Response Normalization** (now replaced by BatchNorm everywhere)

**Architecture:** 5 conv layers + 3 FC layers. 60M parameters. Input 227×227.

**Used today:** Not directly. But AlexNet's design choices (ReLU, dropout, augmentation, GPU training) became the default recipe for all CNN training.

---

## VGG (2014) — Simonyan & Zisserman, Oxford

**Problem solved:** Demonstrated that **depth** is the key factor — a very simple architecture with small filters but many layers beats shallow networks with large filters.

**The insight:** Two 3×3 conv layers have the same receptive field as one 5×5, but fewer parameters and an extra nonlinearity. Three 3×3 layers ≈ one 7×7. Always use 3×3.

**Architecture:** Uniform blocks of 2-4 × (Conv 3×3 → ReLU), followed by MaxPool. Then 3 FC layers (4096 → 4096 → classes).

**Variants:** VGG-11, VGG-13, VGG-16, VGG-19 (number of weight layers). VGG-16 most common.

**Weakness:** FC layers contain 122M of its 138M total parameters. Enormous memory footprint, slow inference.

**Used today:** Rarely as a classifier. Still used as a **perceptual loss network** in image generation and style transfer — its intermediate features are good general-purpose image descriptors.

---

## GoogLeNet / Inception (2014) — Szegedy et al., Google

**Problem solved:** Instead of asking "how deep?", asked "what if we process at multiple scales simultaneously?" Also needed to be computationally efficient to run in production.

**The Inception Module:**

```
Input
  ├── 1×1 conv ──────────────────────────────────────────────┐
  ├── 1×1 conv → 3×3 conv ────────────────────────────────── ┼── Concatenate
  ├── 1×1 conv → 5×5 conv ────────────────────────────────── ┤
  └── MaxPool  → 1×1 conv ────────────────────────────────── ┘
```

Each branch detects patterns at different scales. The 1×1 convolutions before 3×3 and 5×5 **reduce channels first** (bottleneck), making multi-scale processing cheap.

**Architecture:** 22 layers deep with Inception modules stacked. No large FC layers — uses global average pooling before classifier. Only 5M parameters (vs VGG's 138M).

**Also introduced:** **Auxiliary classifiers** — intermediate loss heads partway through the network to help gradient flow to early layers (before ResNet made this unnecessary).

**Inception V3/V4:** Later versions factorized large convolutions further (e.g., 5×5 → two 3×3 in sequence, or 3×3 → 1×3 + 3×1).

**Used today:** Inception V3 is still used as the backbone for computing **FID scores** (image generation quality). InceptionResNet-V2 occasionally in production. Mostly superseded by ResNet/EfficientNet.

---

## DenseNet (2017) — Huang et al.

**Problem solved:** Pushed skip connections to the maximum — instead of adding one shortcut per block (ResNet), **every layer connects directly to every subsequent layer**.

```
ResNet:    x → L1 → (L1+x) → L2 → (L2+L1+x) ... only adjacent skip
DenseNet:  x → L1 → L2(x,L1) → L3(x,L1,L2) → L4(x,L1,L2,L3) → ...
```

Each layer receives the **concatenation** of all previous feature maps as input (not sum like ResNet — concatenation). This means all earlier features are available at every layer.

**Benefits:**
- Feature reuse — earlier low-level features (edges, colors) remain available in deep layers
- Strong gradient flow — every layer directly connected to loss
- Parameter efficient — fewer channels needed per layer because features aren't relearned

**Dense Block structure:**

```
x₀ ──────────────────────────────────────────────┐
x₀ → BN→ReLU→Conv → x₁                          │
[x₀,x₁] → BN→ReLU→Conv → x₂                    │ All fed into
[x₀,x₁,x₂] → BN→ReLU→Conv → x₃                │ next layer
...                                               │
[x₀...xₙ] → Transition layer (1×1 conv + pool) ─┘
```

**Variants:** DenseNet-121, 169, 201, 264 (number of layers).

**Used today:** Medical image segmentation and classification (strong when labeled data is scarce — feature reuse reduces the need for data). Less common in general CV — ResNet and ConvNeXt typically outperform it at scale.

---

## MobileNet (2017) — Howard et al., Google

**Problem solved:** Run CNNs efficiently on mobile and edge devices with limited compute and memory.

**Key innovation:** Replace standard convolutions with **depthwise separable convolutions** — split the convolution into spatial filtering (depthwise, one filter per channel) + channel mixing (pointwise, 1×1):

```
Standard Conv: in_C × out_C × K × K params
Depthwise Sep: in_C × K × K + in_C × out_C params

Reduction factor ≈ 1/out_C + 1/K² ≈ 8-9× fewer params and FLOPs
```

**MobileNetV2:** Added inverted residuals (expand then contract, like ConvNeXt) + linear bottlenecks.

**MobileNetV3:** Added hard-swish activation, squeeze-and-excitation blocks, neural architecture search.

**Used today:** Embedded systems, mobile apps, real-time inference on device. The standard backbone when you need CNNs under 10MB or <100M FLOPs.

---

## EfficientNet (2019) — Tan & Le, Google Brain

**Problem solved:** Given a budget to make a CNN bigger, what's the optimal tradeoff between depth (more layers), width (more channels), and resolution (larger input)?

**The compound scaling coefficient:** Scale all three dimensions together with a fixed ratio:

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ

where α·β²·γ² ≈ 2  (doubling FLOPs)
```

The optimal ratios (α=1.2, β=1.1, γ=1.15) were found by neural architecture search on a small baseline model (EfficientNet-B0), then the compound coefficient φ was scaled up.

**Model series:**

| Model | Params | FLOPs | Top-1 |
|-------|--------|-------|-------|
| B0 | 5.3M | 0.4B | 77.1% |
| B1 | 7.8M | 0.7B | 79.1% |
| B4 | 19M  | 4.2B | 82.6% |
| B7 | 66M  | 37B  | 84.3% |

**Used today:** When you need a strong pretrained backbone but have FLOP or parameter budget constraints. EfficientNet-B4 is a common sweet spot. Being displaced by ConvNeXt which achieves similar accuracy with simpler design and faster inference.

---

## Quick Comparison

| Architecture | Year | Params | Top-1 | Key Idea |
|-------------|------|--------|-------|----------|
| LeNet | 1998 | 0.06M | ~99% MNIST | Conv+Pool blueprint |
| AlexNet | 2012 | 60M | 63.3% | Deep + ReLU + GPU |
| VGG-16 | 2014 | 138M | 74.4% | All 3×3, just go deep |
| GoogLeNet | 2014 | 5M | 74.8% | Multi-scale Inception |
| ResNet-50 | 2015 | 25M | 76.1% | Skip connections ⭐ |
| DenseNet-121 | 2017 | 8M | 74.9% | Dense skip connections |
| MobileNetV2 | 2018 | 3.4M | 72.0% | Efficient, mobile |
| EfficientNet-B4 | 2019 | 19M | 82.6% | Compound scaling |
| ConvNeXt-T | 2022 | 29M | 82.1% | Modern CNN ⭐ |