# Audio Normalization + Feature Preparation

Audio normalization operates at two levels: the raw waveform before feature extraction, and the spectrogram after it. Both matter for stable training.

---

## 1. Waveform-Level Normalization

Applied to the raw audio signal before any transformation.

### Peak Normalization
Scale the waveform so the maximum absolute amplitude equals 1.0.

```python
import numpy as np

def peak_normalize(waveform):
    peak = np.max(np.abs(waveform))
    if peak > 0:
        return waveform / peak
    return waveform

waveform_normalized = peak_normalize(waveform)
# range: [-1.0, 1.0]
```

### RMS Normalization (Loudness Normalization)
Normalize by Root Mean Square energy — makes all audio clips roughly the same perceived loudness.

```python
def rms_normalize(waveform, target_rms=0.1):
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 0:
        return waveform * (target_rms / rms)
    return waveform
```

### Resampling
All audio in a dataset must share the same sample rate.

```python
import librosa

# Load and resample to 16000 Hz (speech standard)
waveform, sr = librosa.load("audio.wav", sr=16000)

# Or resample after loading
waveform_resampled = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
```

---

## 2. Spectrogram-Level Normalization

Applied to the Mel-Spectrogram or STFT output before feeding into the model.

### dB Conversion (Log Compression)
The most important normalization step for spectrograms. Raw power values span many orders of magnitude — log compression maps them to a perceptually linear, numerically manageable range.

```python
import librosa
import numpy as np

mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)

# Power → dB  (log compression)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
# Values now in range approximately [-80, 0] dB

# Or for magnitude spectrogram
mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
```

### Global Mean/Std Normalization (Z-score)
Standardize the spectrogram to zero mean and unit variance — same principle as tabular standardization.

```python
# Per-sample normalization (most common)
def normalize_spectrogram(spec):
    mean = spec.mean()
    std  = spec.std()
    return (spec - mean) / (std + 1e-8)   # 1e-8 for numerical stability

mel_normalized = normalize_spectrogram(mel_db)
```

### Dataset-Level Normalization
Compute mean and std across the entire training set, then apply to each sample.

```python
# Compute dataset statistics
all_specs = []
for audio_file in train_files:
    waveform, sr = librosa.load(audio_file, sr=16000)
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    all_specs.append(mel_db)

all_specs = np.concatenate(all_specs, axis=1)   # (n_mels, total_frames)
global_mean = all_specs.mean(axis=1, keepdims=True)
global_std  = all_specs.std(axis=1, keepdims=True)

# Apply to each sample
mel_normalized = (mel_db - global_mean) / (global_std + 1e-8)
```

---

## 3. Audio Data Augmentation

Applied to the waveform or spectrogram during training to improve robustness.

### Waveform-Level Augmentations

```python
# Time shifting — shift audio left or right
def time_shift(waveform, shift_max=0.2, sr=16000):
    shift = int(np.random.uniform(-shift_max, shift_max) * sr)
    return np.roll(waveform, shift)

# Pitch shifting
waveform_pitched = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=2)

# Time stretching (without changing pitch)
waveform_stretched = librosa.effects.time_stretch(waveform, rate=1.2)

# Adding background noise
noise = np.random.randn(len(waveform)) * 0.005
waveform_noisy = waveform + noise
```

### Spectrogram-Level Augmentations

**SpecAugment** — the dominant augmentation for speech models (used in Whisper, wav2vec 2.0).

Masks random blocks of time steps and frequency bands directly on the spectrogram.

```python
import torchaudio.transforms as T

# Frequency masking — mask F consecutive frequency bins
freq_mask = T.FrequencyMasking(freq_mask_param=30)
mel_augmented = freq_mask(mel_spec)

# Time masking — mask T consecutive time steps
time_mask = T.TimeMasking(time_mask_param=100)
mel_augmented = time_mask(mel_augmented)

# Full SpecAugment policy (multiple masks)
specaugment = T.Compose([
    T.FrequencyMasking(freq_mask_param=27),
    T.FrequencyMasking(freq_mask_param=27),
    T.TimeMasking(time_mask_param=100, p=1.0),
    T.TimeMasking(time_mask_param=100, p=1.0),
])
```

---

## 4. Padding & Truncation for Audio

Audio clips vary in length — models need fixed-length inputs.

```python
def pad_or_truncate(waveform, target_length, sr=16000):
    current_length = len(waveform)
    if current_length < target_length:
        # Pad with zeros (silence)
        pad_length = target_length - current_length
        waveform = np.pad(waveform, (0, pad_length), mode="constant")
    else:
        # Truncate
        waveform = waveform[:target_length]
    return waveform

# Example: fix all audio to 3 seconds
target = 3 * 16000   # 48,000 samples
waveform_fixed = pad_or_truncate(waveform, target)
```

---

## 5. Complete Preprocessing Pipeline

```python
import librosa
import numpy as np
import torchaudio.transforms as T
import torch

def preprocess_audio(filepath, sr=16000, n_mels=80, n_fft=400, hop_length=160):
    # 1. Load and resample
    waveform, orig_sr = librosa.load(filepath, sr=sr, mono=True)

    # 2. Peak normalize waveform
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

    # 3. Pad or truncate to fixed length
    waveform = pad_or_truncate(waveform, target_length=sr * 5)  # 5 seconds

    # 4. Compute Mel-Spectrogram
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # 5. Log compression (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 6. Normalize spectrogram
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    # 7. Convert to tensor and add channel dim
    tensor = torch.FloatTensor(mel_db).unsqueeze(0)  # (1, n_mels, T)

    return tensor

tensor = preprocess_audio("speech.wav")
print(tensor.shape)   # torch.Size([1, 80, 500])
```

---

## 6. Normalization Summary

| Step | What | Why |
|---|---|---|
| Peak Normalize | waveform / max | Consistent amplitude across files |
| Resample | resample to target sr | Uniform time resolution |
| dB Conversion | power_to_db | Log-compress wide dynamic range |
| Z-score (per sample) | (x - mean) / std | Zero-centered input to model |
| Z-score (dataset) | global mean/std | Consistent scale across dataset |
| SpecAugment | mask freq/time bands | Robustness, prevents overfitting |