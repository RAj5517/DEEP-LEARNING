# ANN Variants

---

## Autoencoder

### What It Is
An MLP trained to compress input into a small **latent representation** (encoding) and then reconstruct the original input from it. No labels needed — the input is both the input and the target.

```
Input (784)  →  Encoder  →  Latent (32)  →  Decoder  →  Output (784)
    x        →  [MLP]   →      z         →   [MLP]   →     x̂

Loss = ||x - x̂||²   (reconstruction loss)
```

The bottleneck forces the network to learn the most important features of the data — if you can reconstruct from 32 numbers what originally had 784, those 32 numbers must capture the essential structure.

**Used for:**
- Dimensionality reduction (alternative to PCA, but nonlinear)
- Anomaly detection: train on normal data, high reconstruction error = anomaly
- Pretraining features before fine-tuning on labeled data

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 32)                  # Bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),  nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()  # Reconstruct 0-1 pixels
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

---

### Variational Autoencoder (VAE)

Standard autoencoders map each input to a single point in latent space. VAEs map each input to a **distribution** (mean + variance) in latent space, then sample from it. This makes the latent space continuous and structured, enabling generation.

```
Input → Encoder → μ, σ → Sample z = μ + σ·ε (ε ~ N(0,1)) → Decoder → Output

Loss = Reconstruction loss + KL Divergence
                              ↑
                   Keeps latent distribution close to N(0,1)
```

The KL divergence term regularizes the latent space — it forces different inputs to map to overlapping distributions rather than isolated points. This continuity means you can sample random `z` and decode it into a valid new sample.

**Used for:** Image generation, data augmentation, learning disentangled representations.

---

### Denoising Autoencoder

Corrupt the input (add noise, mask pixels), train the network to reconstruct the **clean** original. Forces the network to learn robust features rather than just copying input to output.

```
Input x → add noise → x̃ → Autoencoder → x̂ ≈ x (original, not noisy)
```

This is conceptually the same idea behind BERT's masked language modeling and MAE (Masked Autoencoders for vision) — just applied to pixels instead of tokens.

---

## Siamese Network

### What It Is
Two identical networks (same weights, shared) that each process one input, producing embeddings. A distance function then measures similarity between the two embeddings.

```
Input A → [Network] ─┐
                      ├→ distance(emb_A, emb_B) → similarity score
Input B → [Network] ─┘
         (same weights)
```

The network learns an **embedding space** where similar things are close and different things are far apart.

**Loss functions:**
- **Contrastive loss**: pull similar pairs together, push dissimilar pairs apart
- **Triplet loss**: anchor, positive (same class), negative (different class) — positive closer to anchor than negative

**Used for:**
- Face verification (same person or different?)
- Signature verification
- Few-shot learning (is this new image the same class as the example?)
- Duplicate detection

```python
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        distance = torch.nn.functional.pairwise_distance(emb1, emb2)
        return distance
```

---

## MLP-Mixer (Brief)

A 2021 architecture that applies MLP operations along two dimensions of a patch grid: one MLP mixes information **across spatial locations** (token mixing), another mixes information **across channels** (channel mixing). Proved competitive with CNNs and early ViTs on image classification using only MLPs — no convolutions, no attention.

Mostly of theoretical interest — in practice, ViT and ConvNeXt outperform it. Shows that the inductive biases of CNNs and Transformers aren't strictly necessary for vision.

---

## TabNet (Brief)

An attention-based MLP architecture specifically designed for tabular data. Uses sequential attention to select which features to use at each step, providing both performance and interpretability (you can see which features the model attended to).

Competitive with gradient boosting (XGBoost, LightGBM) on some tabular benchmarks, though gradient boosting still tends to win on most structured data tasks.