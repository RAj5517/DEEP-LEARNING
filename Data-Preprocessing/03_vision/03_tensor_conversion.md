# Tensor Conversion

After resizing, normalizing, and augmenting, the image must be converted into a **tensor** — the data structure that PyTorch and TensorFlow operate on. This step also handles the format differences between image libraries and deep learning frameworks.

---

## 1. The Format Problem

Different libraries store images in different formats:

| Library | Format | Shape | Dtype | Range |
|---|---|---|---|---|
| PIL / Pillow | HWC | (H, W, C) | uint8 | [0, 255] |
| OpenCV | HWC | (H, W, C) | uint8 | [0, 255] BGR order |
| NumPy | HWC | (H, W, C) | uint8 / float | [0, 255] |
| PyTorch | CHW | (C, H, W) | float32 | [0.0, 1.0] |
| TensorFlow | HWC | (H, W, C) | float32 | [0.0, 1.0] |

PyTorch uses **CHW** (Channel first). PIL and OpenCV use **HWC** (Channel last). Converting between them is a core part of tensor conversion.

---

## 2. torchvision — transforms.ToTensor()

The standard conversion step in PyTorch pipelines. Does three things in one call:
- PIL Image or NumPy array → `torch.FloatTensor`
- Shape: `(H, W, C)` → `(C, H, W)`
- Pixel range: `[0, 255]` → `[0.0, 1.0]`

```python
from torchvision import transforms
from PIL import Image

img = Image.open("image.jpg")     # shape: (H, W, C), dtype: uint8

to_tensor = transforms.ToTensor()
tensor = to_tensor(img)           # shape: (C, H, W), dtype: float32, range [0,1]

print(tensor.shape)    # torch.Size([3, 224, 224])
print(tensor.dtype)    # torch.float32
print(tensor.min(), tensor.max())  # tensor(0.) tensor(1.)
```

---

## 3. Converting from OpenCV (BGR → RGB)

OpenCV loads images in **BGR** channel order. Must be converted to RGB before passing to models pretrained on RGB data.

```python
import cv2
import torch
import numpy as np

img_bgr = cv2.imread("image.jpg")           # (H, W, 3) BGR uint8
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR → RGB

# Convert to tensor manually
tensor = torch.from_numpy(img_rgb)          # (H, W, C) uint8
tensor = tensor.permute(2, 0, 1)            # → (C, H, W)
tensor = tensor.float() / 255.0            # → float32 [0, 1]
```

---

## 4. Converting from NumPy

```python
import numpy as np
import torch

img_np = np.random.uint8(np.random.randint(0, 255, (224, 224, 3)))

# Manual conversion
tensor = torch.from_numpy(img_np.copy())     # copy() avoids non-writable array issues
tensor = tensor.permute(2, 0, 1).float() / 255.0

# Or use torchvision
from torchvision.transforms.functional import to_tensor
tensor = to_tensor(img_np)
```

---

## 5. Adding a Batch Dimension

Models expect a batch of images, not a single image. Add the batch dimension with `unsqueeze(0)`.

```python
# Single image inference
tensor = transforms.ToTensor()(image)    # shape: (C, H, W)
batch  = tensor.unsqueeze(0)             # shape: (1, C, H, W)

# Batch of images from DataLoader — already batched
for images, labels in dataloader:
    # images shape: (B, C, H, W)
    outputs = model(images)
```

---

## 6. TensorFlow / Keras Conversion

TensorFlow uses HWC format (channel last) by default.

```python
import tensorflow as tf
from PIL import Image
import numpy as np

img = Image.open("image.jpg").resize((224, 224))
img_array = np.array(img)                      # (H, W, C) uint8

# Convert to float tensor
tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
tensor = tensor / 255.0                        # normalize to [0, 1]

# Add batch dimension
batch = tf.expand_dims(tensor, axis=0)         # (1, H, W, C)

# Using Keras preprocessing
tensor = tf.keras.applications.resnet50.preprocess_input(img_array)
```

---

## 7. Moving Tensors to GPU

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move single tensor
tensor = tensor.to(device)

# Move model and data together
model  = model.to(device)
images = images.to(device)
labels = labels.to(device)

# Check device
print(tensor.device)   # device(type='cuda', index=0)
```

---

## 8. dtype Considerations

| dtype | Bits | Use Case |
|---|---|---|
| `float32` | 32 | Default training precision |
| `float16` | 16 | Mixed precision training (AMP) — faster on modern GPUs |
| `bfloat16` | 16 | Training on TPUs and A100/H100 GPUs |
| `uint8` | 8 | Storage only — never feed directly to model |

```python
# Mixed precision (automatic — recommended for modern GPUs)
from torch.cuda.amp import autocast

with autocast():
    outputs = model(images)   # runs in float16 where safe, float32 elsewhere
```

---

## 9. Complete Pipeline

```python
from torchvision import transforms
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),                           # HWC uint8 → CHW float [0,1]
    transforms.Normalize(                            # channel-wise standardization
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

img    = Image.open("image.jpg").convert("RGB")
tensor = transform(img)          # shape: (3, 224, 224)
batch  = tensor.unsqueeze(0)     # shape: (1, 3, 224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch  = batch.to(device)

with torch.no_grad():
    output = model(batch)
```