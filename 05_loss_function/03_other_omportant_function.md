## 🟡 Very Important (Modern DL)

These are essential for specific domains but you'll encounter them frequently in modern architectures.

---

### 9. Smooth L1 Loss / Huber Loss

**Category:** Regression  
**Status:** ⭐ Standard for Object Detection

#### Mathematical Definition
```
Smooth L1:
L(x) = { 0.5·x²         if |x| < 1
       { |x| - 0.5      otherwise

Huber Loss (generalized):
L_δ(x) = { 0.5·x²/δ           if |x| ≤ δ
         { |x| - 0.5·δ        otherwise
```

#### Key Characteristics
- **Hybrid**: Quadratic for small errors, linear for large
- **Robust**: Less sensitive to outliers than MSE
- **Smooth gradients**: Unlike pure L1 (MAE)
- **Tunable threshold**: δ parameter controls transition point

#### When to Use
✅ Use in: Object detection (Faster R-CNN, YOLO), Robust regression, RL (TD error)  
❌ Avoid in: Standard regression (if MSE works), When all errors are similar scale

#### Code Example
```python
import torch.nn as nn

# Smooth L1 (δ=1.0)
criterion = nn.SmoothL1Loss(beta=1.0)

# Huber Loss
criterion = nn.HuberLoss(delta=1.0)
```

#### Practical Tips
- Faster R-CNN uses β=1.0 for bounding box regression
- Tune δ based on your error scale
- More stable than MSE for detection tasks

---

### 10. CTC Loss (Connectionist Temporal Classification)

**Category:** Sequence Modeling  
**Status:** ⭐ Standard for Speech Recognition & OCR

#### Mathematical Definition
```
L_CTC = -log P(y|x) = -log Σ_{π∈B⁻¹(y)} Π_t p_t(π_t|x)

where:
  B⁻¹(y) = all possible alignments that map to label sequence y
  π      = alignment path (with blanks)
  p_t    = probability at time step t
```

#### Key Characteristics
- **Alignment-free**: No need for frame-level labels
- **Handles variable length**: Input/output can differ in length
- **Introduces blank token**: Allows repetitions and silence
- **Dynamic programming**: Uses forward-backward algorithm

#### When to Use
✅ Use in: Speech recognition (Whisper), OCR, Handwriting recognition, Any sequence alignment task  
❌ Avoid in: Fixed-length sequences, When alignment is available, Standard NLP (use CE)

#### Code Example
```python
import torch.nn as nn

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')

# log_probs: [T, N, C] - T: time steps, N: batch, C: classes
# targets: [N, S] - target sequences
# input_lengths: [N] - length of each input
# target_lengths: [N] - length of each target

loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

#### Practical Tips
- Used in Wav2Vec, Whisper (older versions), Tesseract OCR
- Requires monotonic alignment assumption
- Can suffer from "peaky" distributions (add label smoothing)
- Combine with language model for best results

---

### 11. Perceptual Loss

**Category:** Image Quality  
**Status:** 🔥 Essential for Image Generation & Enhancement

#### Mathematical Definition
```
L_perceptual = Σ_l ||φ_l(ŷ) - φ_l(y)||²

where:
  φ_l = features from layer l of pre-trained network (VGG, etc.)
  ŷ   = generated image
  y   = target image
```

#### Key Characteristics
- **Feature-based**: Compares high-level features, not pixels
- **Pre-trained network**: Usually VGG-16 or VGG-19
- **Multi-scale**: Can use multiple layers
- **Perceptually aligned**: Better matches human judgment

#### When to Use
✅ Use in: Style transfer, Super-resolution, Image-to-image translation, Denoising  
❌ Avoid in: When pixel accuracy critical, Very low-level tasks, No pre-trained models available

#### Code Example
```python
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3']):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.model = nn.Sequential()
        
        # Extract features up to specified layers
        layer_map = {
            'conv1_2': 4, 'conv2_2': 9, 
            'conv3_3': 16, 'conv4_3': 23
        }
        
        for name, layer_idx in layer_map.items():
            if name in layers:
                self.model.add_module(name, vgg[:layer_idx])
        
        # Freeze VGG parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, generated, target):
        gen_features = self.model(generated)
        target_features = self.model(target)
        
        loss = nn.functional.mse_loss(gen_features, target_features)
        return loss
```

#### Practical Tips
- Combine with pixel loss: `λ₁·L_pixel + λ₂·L_perceptual`
- VGG layers: early (texture), middle (structure), late (semantics)
- Normalize inputs to ImageNet statistics
- Can add style loss (Gram matrices) for style transfer

---

### 12. Triplet Loss

**Category:** Metric Learning  
**Status:** ⭐ Face Recognition, Person Re-ID

#### Mathematical Definition
```
L = max(0, D(a,p) - D(a,n) + margin)

where:
  a = anchor
  p = positive (same class as anchor)
  n = negative (different class)
  D = distance function (L2, cosine)
```

#### When to Use
✅ Use in: Face recognition, Person re-identification, Signature verification  
❌ Avoid in: Standard classification, When sampling triplets is hard

#### Code Example
```python
import torch.nn as nn

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)  # L2 distance
loss = triplet_loss(anchor, positive, negative)
```

#### Practical Tips
- Hard negative mining crucial for good performance
- Margin typically 0.2-1.0
- Semi-hard negatives work best (harder than easy, easier than hard)
- Use with online mining during training

---

### 13. IoU Loss (and variants)

**Category:** Object Detection  
**Status:** 🔥 State-of-art for Bounding Box Regression

#### Mathematical Definition
```
IoU = Area(A ∩ B) / Area(A ∪ B)

IoU Loss = 1 - IoU

GIoU = IoU - |C \ (A ∪ B)| / |C|  (C: smallest enclosing box)
DIoU = IoU - ρ²(a,b) / c²         (distance penalty)
CIoU = DIoU - α·v                  (aspect ratio penalty)
```

#### When to Use
✅ Use in: YOLO, object detection, Instance segmentation  
❌ Avoid in: Non-detection tasks, When boxes don't overlap (GIoU handles this)

#### Code Example
```python
def iou_loss(pred_boxes, target_boxes):
    # pred_boxes, target_boxes: [N, 4] (x1, y1, x2, y2)
    
    # Intersection
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    # Union
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])
    union = area_pred + area_target - intersection
    
    iou = intersection / (union + 1e-6)
    return 1 - iou.mean()
```

#### Practical Tips
- GIoU: Better for non-overlapping boxes
- DIoU: Faster convergence
- CIoU: Best overall (YOLOv5 default)
- Combine with classification loss

---

### 14. Hinge Loss

**Category:** SVM, GANs  
**Status:** 🔥 Modern GAN Training

#### Mathematical Definition
```
Binary SVM:
L = max(0, 1 - y·f(x))  where y ∈ {-1, 1}

Hinge Loss for GANs:
L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
L_G = -E[D(G(z))]
```

#### When to Use
✅ Use in: StyleGAN, BigGAN, SVMs  
❌ Avoid in: Standard classification (use CE)

#### Code Example
```python
# For GAN discriminator
def hinge_loss_dis(real_logits, fake_logits):
    loss_real = torch.mean(F.relu(1. - real_logits))
    loss_fake = torch.mean(F.relu(1. + fake_logits))
    return loss_real + loss_fake

# For GAN generator
def hinge_loss_gen(fake_logits):
    return -torch.mean(fake_logits)
```

#### Practical Tips
- Popular in high-quality image synthesis (StyleGAN2)
- More stable than BCE for some architectures
- No sigmoid needed in discriminator output

---