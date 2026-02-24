# Dimensionality Reduction

Dimensionality reduction compresses high-dimensional data into a lower-dimensional representation while preserving the most important structure. Unlike feature selection (which keeps original features), dimensionality reduction creates new features that are combinations of the originals.

---

## 1. When to Use

- Hundreds or thousands of features after engineering
- Visualization of high-dimensional data (reduce to 2D/3D)
- Features are highly correlated — much redundancy exists
- Distance-based models failing due to curse of dimensionality
- Speeding up downstream model training

---

## 2. PCA — Principal Component Analysis

PCA finds the directions (principal components) of maximum variance in the data and projects the data onto those directions. It is a linear transformation.

### How It Works

```
1. Standardize features (zero mean, unit variance)
2. Compute covariance matrix
3. Compute eigenvectors (principal components) and eigenvalues
4. Sort eigenvectors by eigenvalue (variance explained) descending
5. Project data onto top K eigenvectors
```

### Implementation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# CRITICAL: always standardize before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Step 1: Fit PCA to find how many components to keep
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(cumulative_variance)
plt.axhline(y=0.95, color="r", linestyle="--", label="95% variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA — How Many Components?")
plt.legend()
plt.show()

# Step 2: Choose n_components that explain 95% of variance
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components}")

# Step 3: Apply PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_scaled)
X_test_pca  = pca.transform(scaler.transform(X_test))  # fit on train only

print(f"Original shape: {X_train.shape}")
print(f"Reduced shape:  {X_train_pca.shape}")
```

### Choosing n_components

```python
# Option 1: Fixed number
pca = PCA(n_components=50)

# Option 2: Variance threshold (recommended)
pca = PCA(n_components=0.95)   # keep enough components for 95% variance

# Option 3: Check component contributions
for i, var in enumerate(pca_full.explained_variance_ratio_[:10]):
    print(f"PC{i+1}: {var:.3f}  ({cumulative_variance[i]:.3f} cumulative)")
```

### Visualizing in 2D

```python
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
for label in np.unique(y):
    mask = y == label
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=str(label), alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA — 2D Projection")
plt.legend()
plt.show()
```

### Limitations of PCA

- **Linear only** — cannot capture non-linear structure (a spiral, manifold, etc.)
- **Assumes variance = information** — may discard low-variance dimensions that are discriminative
- **Not interpretable** — principal components are mixtures of original features
- **Sensitive to outliers** — use `PCA(svd_solver="randomized")` or robust PCA for outlier-heavy data

---

## 3. Kernel PCA

Extends PCA to non-linear relationships using the kernel trick.

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(
    n_components = 50,
    kernel       = "rbf",    # rbf, poly, sigmoid, cosine
    gamma        = 0.1,
    fit_inverse_transform = True
)
X_kpca = kpca.fit_transform(X_scaled)
```

---

## 4. t-SNE — Visualization Only

t-SNE produces stunning 2D/3D visualizations of cluster structure. **Not for preprocessing** — it is non-deterministic, cannot transform new data, and is only for exploration.

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components = 2,
    perplexity   = 30,       # controls local vs global structure (5–50)
    n_iter       = 1000,
    random_state = 42
)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.colorbar(scatter)
plt.title("t-SNE Visualization")
plt.show()
```

---

## 5. UMAP — Better Than t-SNE for Most Cases

Faster than t-SNE, preserves global structure better, and can transform new data.

```python
import umap

reducer = umap.UMAP(
    n_components = 2,       # or higher (10–50) for preprocessing
    n_neighbors  = 15,      # local neighborhood size
    min_dist     = 0.1,     # how tightly to pack points
    metric       = "euclidean",
    random_state = 42
)
X_umap = reducer.fit_transform(X_scaled)

# Can transform new data (unlike t-SNE)
X_test_umap = reducer.transform(scaler.transform(X_test))
```

---

## 6. Autoencoders — Neural Network Dimensionality Reduction

An autoencoder learns a compressed **latent representation** of the data through a bottleneck layer. It is the non-linear, neural network equivalent of PCA.

### Architecture

```
Input (D) → Encoder → Bottleneck (d << D) → Decoder → Reconstructed Input (D)

Loss = Reconstruction Error = MSE(input, output)
```

The encoder half is used as the dimensionality reducer.

### Basic Autoencoder

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)   # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = Autoencoder(input_dim=X.shape[1], latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training
for epoch in range(100):
    for x_batch in dataloader:
        x_batch = x_batch.to(device)
        recon, z = model(x_batch)
        loss = criterion(recon, x_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Extract compressed representations
model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    _, Z = model(X_tensor)
    Z = Z.cpu().numpy()   # shape: (N, latent_dim)
```

### Variational Autoencoder (VAE)

VAEs learn a probabilistic latent space — smoother and more structured than standard autoencoders.

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.mu      = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h   = self.encoder(x)
        mu, log_var = self.mu(h), self.log_var(h)
        z   = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# VAE loss = reconstruction loss + KL divergence
def vae_loss(recon, x, mu, log_var):
    recon_loss = nn.MSELoss()(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss
```

---

## 7. Comparison Table

| Method | Linear | New Data | Visualization | Use Case |
|---|---|---|---|---|
| PCA | Yes | Yes | 2D/3D | General preprocessing, correlated features |
| Kernel PCA | No | Yes | 2D/3D | Non-linear structure |
| t-SNE | No | No | 2D/3D only | Exploration only |
| UMAP | No | Yes | 2D–50D | Visualization + preprocessing |
| Autoencoder | No | Yes | Any dim | Complex non-linear compression |
| VAE | No | Yes | Any dim | Generative tasks + compression |

---

## 8. Decision Framework

```
Is the relationship between features mostly linear?
├── YES → PCA  (fast, interpretable, reliable)
└── NO
    ├── Need to transform new test data?
    │   ├── YES → UMAP (n_components > 2) or Autoencoder
    │   └── NO  → t-SNE or UMAP (visualization only)
    └── Have enough data to train a neural network?
        ├── YES → Autoencoder or VAE
        └── NO  → Kernel PCA
```