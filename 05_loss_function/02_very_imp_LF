---

## 🔴 Absolute Must-Know (Top 8)

These are the foundation of deep learning. You'll encounter them in almost every project.

---

### 1. MSE (Mean Squared Error)

**Category:** Regression  
**Status:** ⭐ Industry Standard for Regression

#### Mathematical Definition
```
L_MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

where:
  n  = number of samples
  yᵢ = actual value
  ŷᵢ = predicted value
```

#### Key Characteristics
- **Quadratic penalty**: Errors are squared, so large errors are heavily penalized
- **Differentiable**: Smooth gradients everywhere (great for optimization)
- **Outlier sensitive**: A single large error can dominate the loss
- **Assumes Gaussian noise**: Optimal under normal error distribution

#### When to Use
✅ **Use when:**
- Default regression task with well-behaved data
- You want smooth gradients
- Large errors should be penalized more than small ones
- Data doesn't have many outliers

❌ **Avoid when:**
- Data has significant outliers
- Error distribution is non-Gaussian
- You need robust predictions

#### Code Example
```python
import torch
import torch.nn as nn

# PyTorch
criterion = nn.MSELoss()
output = model(input)
loss = criterion(output, target)

# Manual implementation
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
```

#### Practical Tips
- Scale your targets to similar ranges (normalization helps)
- Consider using MAE or Huber if you have outliers
- MSE in pixels² for images, so prefer SSIM or Perceptual loss for image quality

---

### 2. Cross Entropy (Categorical)

**Category:** Classification (Multi-class)  
**Status:** ⭐ Gold Standard for Classification

#### Mathematical Definition
```
L_CE = -Σᵢ yᵢ · log(ŷᵢ)

For single sample with C classes:
L = -log(ŷ_c)  where c is the true class

where:
  yᵢ = true probability (one-hot: 0 or 1)
  ŷᵢ = predicted probability (from softmax)
```

#### Key Characteristics
- **Logarithmic penalty**: Heavily penalizes confident wrong predictions
- **Requires probability outputs**: Use softmax activation
- **Mutually exclusive classes**: Only one class can be correct
- **Maximum likelihood**: Equivalent to negative log-likelihood

#### When to Use
✅ **Use when:**
- Multi-class classification (3+ classes)
- Each sample belongs to exactly ONE class
- Using neural networks with softmax output

❌ **Avoid when:**
- Binary classification (use BCE instead)
- Multi-label problems (multiple classes can be true)
- Extremely imbalanced datasets (consider Focal Loss)

#### Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch (includes softmax)
criterion = nn.CrossEntropyLoss()
logits = model(input)  # Raw scores, no softmax needed
loss = criterion(logits, target)  # target is class index

# With label smoothing (regularization)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Manual implementation
def cross_entropy(logits, target):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs[range(len(target)), target].mean()
    return loss
```

#### Practical Tips
- Always use `nn.CrossEntropyLoss()` instead of manual softmax + log
- Add label smoothing (0.1) to prevent overconfidence
- Use class weights for imbalanced datasets
- Monitor per-class accuracy, not just loss

---

### 3. BCE (Binary Cross Entropy)

**Category:** Classification (Binary)  
**Status:** ⭐ Standard for Binary Classification

#### Mathematical Definition
```
L_BCE = -(1/n) Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

where:
  yᵢ ∈ {0, 1} = true label
  ŷᵢ ∈ [0, 1] = predicted probability
```

#### Key Characteristics
- **Two-class specialization**: Optimized version of cross entropy for binary case
- **Requires sigmoid output**: Predictions must be in [0, 1]
- **Symmetric**: Treats both classes equally (unless weighted)

#### When to Use
✅ **Use when:**
- Binary classification (yes/no, true/false)
- Multi-label classification (each label independently)
- Output is probability between 0 and 1

❌ **Avoid when:**
- More than 2 mutually exclusive classes
- Working with logits (use BCEWithLogitsLoss instead)

#### Code Example
```python
import torch
import torch.nn as nn

# Standard BCE (requires sigmoid output)
criterion = nn.BCELoss()
output = torch.sigmoid(model(input))
loss = criterion(output, target)

# Better: BCE with Logits (numerically stable)
criterion = nn.BCEWithLogitsLoss()
logits = model(input)  # Raw scores
loss = criterion(logits, target)

# With positive class weighting
pos_weight = torch.tensor([2.0])  # Penalize false negatives 2x more
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

#### Practical Tips
- **Always use `BCEWithLogitsLoss`** instead of sigmoid + BCE (more stable)
- Use `pos_weight` for imbalanced datasets
- For multi-label: apply BCE independently to each label
- Consider Focal Loss if severe class imbalance exists

---

### 4. Focal Loss

**Category:** Classification (Imbalanced Data)  
**Status:** 🔥 Modern Essential for Imbalanced Problems

#### Mathematical Definition
```
L_Focal = -α(1 - ŷ)^γ · log(ŷ)

where:
  α     = class balancing weight (typically 0.25)
  γ     = focusing parameter (typically 2.0)
  ŷ     = predicted probability for true class
  (1-ŷ)^γ = modulating factor (reduces loss for well-classified)
```

#### Key Characteristics
- **Down-weights easy examples**: Focuses on hard negatives
- **Up-weights rare classes**: Handles severe imbalance (1:1000)
- **Derived from Cross Entropy**: With dynamic scaling
- **Two hyperparameters**: α for class balance, γ for focusing

#### When to Use
✅ **Use when:**
- Extreme class imbalance (e.g., 99% negative class)
- Object detection (most anchors are background)
- Medical diagnosis (rare diseases)
- Fraud detection (rare positive cases)

❌ **Avoid when:**
- Balanced datasets (adds unnecessary complexity)
- You haven't tried weighted CE first
- Training is unstable (reduce γ)

#### Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Binary focal loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        p_t = torch.exp(-bce_loss)  # Predicted probability
        focal_loss = self.alpha * (1 - p_t)**self.gamma * bce_loss
        return focal_loss.mean()

# Multi-class focal loss
class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t)**self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss *= self.alpha[targets]
        
        return focal_loss.mean()

# Usage
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, targets)
```

#### Hyperparameter Tuning
- **γ (gamma):**
  - γ=0 → Standard Cross Entropy
  - γ=2 → Default, works well for most cases
  - γ=5 → Very strong focus on hard examples
  
- **α (alpha):**
  - α=0.25 for binary (favors rare class)
  - α=[w₁, w₂, ...] for multi-class weights

#### Practical Tips
- Start with γ=2.0, α=0.25 (proven defaults from RetinaNet)
- If training is unstable, reduce γ to 1.0 or 1.5
- Combine with class weights (α) for best results
- Monitor both easy and hard example losses separately
- Used in: RetinaNet, YOLO variants, many modern detectors

---

### 5. Dice Loss

**Category:** Segmentation  
**Status:** ⭐ Standard for Medical Imaging & Segmentation

#### Mathematical Definition
```
Dice Coefficient = 2·|A ∩ B| / (|A| + |B|)

Dice Loss = 1 - Dice Coefficient

          = 1 - (2·Σᵢ yᵢ·ŷᵢ + ε) / (Σᵢ yᵢ + Σᵢ ŷᵢ + ε)

where:
  A, B = predicted and ground truth sets
  yᵢ   = ground truth pixel (0 or 1)
  ŷᵢ   = predicted probability [0, 1]
  ε    = smoothing term (prevents division by zero)
```

#### Key Characteristics
- **Overlap-based**: Measures intersection over union-like metric
- **Class imbalance robust**: Works well when object is small
- **Differentiable**: Soft approximation of Dice coefficient
- **Range [0, 1]**: 0 = perfect overlap, 1 = no overlap

#### When to Use
✅ **Use when:**
- Semantic/medical image segmentation
- Object occupies small portion of image
- Pixel-level class imbalance (e.g., tumor vs. background)
- Care more about object shape than per-pixel accuracy

❌ **Avoid when:**
- Balanced segmentation tasks
- Need pixel-wise precision
- Used alone (combine with CE for better gradients)

#### Code Example
```python
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # predictions: [B, C, H, W] (after sigmoid/softmax)
        # targets: [B, C, H, W] (one-hot encoded)
        
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice

# Multi-class Dice Loss
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # predictions: [B, C, H, W]
        # targets: [B, H, W] (class indices)
        
        num_classes = predictions.shape[1]
        dice_loss = 0
        
        for class_idx in range(num_classes):
            pred_class = predictions[:, class_idx]
            target_class = (targets == class_idx).float()
            
            intersection = (pred_class * target_class).sum()
            dice = (2. * intersection + self.smooth) / (
                pred_class.sum() + target_class.sum() + self.smooth
            )
            dice_loss += 1 - dice
        
        return dice_loss / num_classes

# Combined loss (common practice)
class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        dice_loss = self.dice(predictions, targets)
        ce_loss = self.ce(predictions, targets)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss
```

#### Practical Tips
- **Always combine with Cross Entropy**: Dice for region, CE for boundaries
- Common combination: `0.5·Dice + 0.5·CE`
- Add smoothing term (ε=1.0) to prevent instability
- For multi-class, compute per-class and average
- Related: IoU Loss, Tversky Loss (generalizations of Dice)

---

### 6. Contrastive Loss / InfoNCE

**Category:** Metric Learning, Self-Supervised Learning  
**Status:** 🔥 Foundation of Modern Embeddings (CLIP, SimCLR)

#### Mathematical Definition

**Contrastive Loss (Siamese Networks):**
```
L = (1-y)·D² + y·max(0, m - D)²

where:
  y = 1 if same class, 0 if different
  D = distance between embeddings
  m = margin (threshold)
```

**InfoNCE (Noise Contrastive Estimation):**
```
L = -log[ exp(sim(a, p)/τ) / Σᵢ exp(sim(a, nᵢ)/τ) ]

where:
  a  = anchor embedding
  p  = positive example (similar)
  nᵢ = negative examples (dissimilar)
  τ  = temperature parameter
  sim = similarity function (cosine, dot product)
```

#### Key Characteristics
- **Learns embeddings**: Maps data to metric space where similar items are close
- **Requires pairs/triplets**: Need positive and negative examples
- **Temperature parameter**: Controls separation hardness (τ)
- **Batch size matters**: More negatives = better contrast

#### When to Use
✅ **Use when:**
- Learning embeddings for similarity search
- Self-supervised learning (SimCLR, MoCo)
- Multi-modal learning (CLIP: image-text)
- Face recognition, person re-identification
- Few-shot learning

❌ **Avoid when:**
- Standard supervised classification (use Cross Entropy)
- Can't generate good positive/negative pairs
- Very small batch sizes (<32)

#### Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# InfoNCE Loss (CLIP-style)
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features_a, features_b):
        # features_a, features_b: [batch_size, embedding_dim]
        # Normalize features
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        batch_size = features_a.shape[0]
        
        # Compute similarity matrix
        logits = features_a @ features_b.T / self.temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric loss (a→b and b→a)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        
        return (loss_a + loss_b) / 2

# SimCLR-style Contrastive Loss
class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        # z_i, z_j: [batch_size, embedding_dim] (two augmented views)
        batch_size = z_i.shape[0]
        
        # Normalize
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        sim = z @ z.T / self.temperature  # [2B, 2B]
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -9e15)
        
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_sim = torch.cat([
            sim[range(batch_size), range(batch_size, 2*batch_size)],
            sim[range(batch_size, 2*batch_size), range(batch_size)]
        ])
        
        # Compute loss
        loss = -pos_sim + torch.logsumexp(sim, dim=-1)
        return loss.mean()

# Usage
criterion = InfoNCELoss(temperature=0.07)
image_features = image_encoder(images)
text_features = text_encoder(texts)
loss = criterion(image_features, text_features)
```

#### Hyperparameter Tuning
- **Temperature (τ):**
  - Lower (0.01-0.1): Harder negatives, sharper separation
  - Higher (0.5-1.0): Softer negatives, smoother learning
  - CLIP uses τ=0.07, SimCLR uses τ=0.5

- **Batch Size:**
  - Larger is better (more negatives)
  - CLIP: 32K, SimCLR: 4K minimum recommended
  - Use gradient accumulation if GPU memory limited

#### Practical Tips
- Large batch sizes are crucial (use distributed training)
- Temperature is critical: tune carefully (0.05-0.5 range)
- Use hard negative mining for better results
- Add momentum encoder for stability (MoCo)
- Works best with strong data augmentation
- Used in: CLIP, SimCLR, MoCo, BYOL, SwAV, DINO

---

### 7. KL Divergence (Kullback-Leibler)

**Category:** Generative Models, Distribution Matching  
**Status:** ⭐ Core Component of VAEs and Distillation

#### Mathematical Definition
```
KL(P || Q) = Σᵢ P(i) · log(P(i) / Q(i))
           = Σᵢ P(i) · [log P(i) - log Q(i)]
           = E_P[log P - log Q]

For Gaussians:
KL(N(μ,σ²) || N(0,1)) = 0.5 · [σ² + μ² - 1 - log(σ²)]
```

#### Key Characteristics
- **Asymmetric**: KL(P||Q) ≠ KL(Q||P)
- **Non-negative**: KL ≥ 0, equals 0 only if P = Q
- **Information measure**: Measures "surprise" when using Q instead of P
- **Not a true metric**: Doesn't satisfy triangle inequality

#### When to Use
✅ **Use when:**
- Variational Autoencoders (VAE) - regularize latent space
- Knowledge distillation - match teacher distribution
- Distribution alignment tasks
- Forcing posterior to match prior

❌ **Avoid when:**
- Need symmetric distance (use JS divergence)
- Distributions have different support (can be infinite)
- Simple regression tasks (use MSE)

#### Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# KL Divergence between distributions
kl_div = nn.KLDivLoss(reduction='batchmean')
# Input: log probabilities, Target: probabilities
loss = kl_div(F.log_softmax(input, dim=1), F.softmax(target, dim=1))

# VAE KL Loss (Gaussian prior)
def vae_kl_loss(mu, logvar):
    """
    KL divergence between N(mu, var) and N(0, 1)
    mu: [batch_size, latent_dim]
    logvar: [batch_size, latent_dim] (log variance for stability)
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()

# Complete VAE Loss
class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # Beta-VAE weighting
    
    def forward(self, recon_x, x, mu, logvar):
        # Reconstruction loss (binary cross entropy for images)
        recon_loss = F.binary_cross_entropy(
            recon_x.view(-1), x.view(-1), reduction='sum'
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Total loss
        return recon_loss + self.beta * kl_loss

# Knowledge Distillation
def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    """
    Soft target distillation using KL divergence
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    kl_loss = F.kl_div(
        student_log_probs, teacher_probs, 
        reduction='batchmean'
    )
    
    # Scale by temperature^2 to match gradients
    return kl_loss * (temperature ** 2)
```

#### Practical Tips
- **VAE Training:**
  - Balance reconstruction vs. KL (use β-VAE if needed)
  - KL annealing: gradually increase weight from 0→1
  - Monitor KL collapse (when KL→0, model ignores latent code)

- **Knowledge Distillation:**
  - Higher temperature (T=3-5) for softer distributions
  - Combine with hard label loss: `α·soft_loss + (1-α)·hard_loss`

- **Numerical Stability:**
  - Use log-space computations
  - Clip values to prevent log(0)
  - In PyTorch, always use `log_softmax` + `softmax`

---

### 8. Wasserstein Distance (Earth Mover's Distance)

**Category:** GAN Training  
**Status:** 🔥 Stable Alternative to Original GAN Loss

#### Mathematical Definition
```
W(P_r, P_g) = inf_{γ∈Π(P_r,P_g)} E_(x,y)~γ[||x - y||]

Practical form (Kantorovich-Rubinstein):
W(P_r, P_g) = sup_{||f||_L ≤ 1} E_x~P_r[f(x)] - E_x~P_g[f(x)]

WGAN Loss:
L_D = E_x~P_r[D(x)] - E_x~P_g[D(G(z))]  (maximize)
L_G = -E_x~P_g[D(G(z))]                  (minimize)
```

#### Key Characteristics
- **Meaningful loss**: Correlates with sample quality throughout training
- **No mode collapse**: More stable than original GAN
- **Requires Lipschitz constraint**: On discriminator (critic)
- **No saturation**: Generator gradients don't vanish

#### When to Use
✅ **Use when:**
- Training GANs (more stable than BCE-based)
- Need interpretable training metric
- Experiencing mode collapse with standard GAN
- Want reliable convergence

❌ **Avoid when:**
- Don't need generative modeling
- Can't enforce Lipschitz constraint properly
- Computational budget is very limited

#### Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# WGAN with Gradient Penalty (WGAN-GP)
class WGANGPLoss:
    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp
    
    def gradient_penalty(self, critic, real_data, fake_data):
        batch_size = real_data.size(0)
        
        # Random interpolation between real and fake
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # Critic scores
        d_interpolates = critic(interpolates)
        
        # Gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Penalty: (||∇|| - 1)²
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty
    
    def discriminator_loss(self, critic, real_data, fake_data):
        # Wasserstein distance
        d_real = critic(real_data).mean()
        d_fake = critic(fake_data).mean()
        
        # Gradient penalty
        gp = self.gradient_penalty(critic, real_data, fake_data)
        
        # Total discriminator loss (minimize negative Wasserstein)
        loss = d_fake - d_real + self.lambda_gp * gp
        return loss
    
    def generator_loss(self, critic, fake_data):
        # Generator wants to maximize critic score on fakes
        return -critic(fake_data).mean()

# Usage in training loop
wgan_loss = WGANGPLoss(lambda_gp=10)

# Discriminator step
fake_data = generator(z)
d_loss = wgan_loss.discriminator_loss(critic, real_data, fake_data.detach())
d_loss.backward()
d_optimizer.step()

# Generator step
fake_data = generator(z)
g_loss = wgan_loss.generator_loss(critic, fake_data)
g_loss.backward()
g_optimizer.step()
```

#### Enforcing Lipschitz Constraint

**Method 1: Weight Clipping (Original WGAN)**
```python
# After each discriminator update
for p in critic.parameters():
    p.data.clamp_(-0.01, 0.01)
```
⚠️ Problematic: Can lead to capacity underuse and gradient issues

**Method 2: Gradient Penalty (WGAN-GP)** ✅ Recommended
```python
# As shown in code above
# Penalizes ||∇_x D(x)|| ≠ 1
```

**Method 3: Spectral Normalization**
```python
from torch.nn.utils import spectral_norm

# Apply to all discriminator layers
self.conv1 = spectral_norm(nn.Conv2d(3, 64, 4, 2, 1))
self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
```

#### Practical Tips
- Use WGAN-GP (gradient penalty) or Spectral Normalization
- Train critic more than generator (5:1 ratio common)
- Don't use batch normalization in critic (breaks Lipschitz)
- Use layer normalization or instance normalization instead
- Monitor Wasserstein distance (should decrease = quality improves)
- Learning rates: lr_critic = 1e-4, lr_gen = 1e-4 or lower
- No sigmoid in critic (outputs unbounded scores)

#### Hyperparameter Tuning
- `lambda_gp = 10`: Standard, increase if gradients explode
- Critic iterations: 5 per generator step (can reduce to 1 after stable)
- Weight clipping: ±0.01 (if using, though GP is better)

---
