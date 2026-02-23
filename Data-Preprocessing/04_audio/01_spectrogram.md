# Spectrogram — Time → Frequency Conversion

A raw waveform tells you **how loud** a sound is at each moment in time. A spectrogram tells you **which frequencies** are present at each moment. This transformation is the foundation of all modern audio deep learning.

---

## 1. Raw Waveform

Audio is stored as a sequence of amplitude samples recorded at a fixed rate.

```
Sample Rate (sr) = 22050 Hz  →  22,050 samples per second
Duration = 3 seconds         →  66,150 samples total
Shape: (66150,)               →  1D array of floats in [-1.0, 1.0]
```

```python
import librosa
import soundfile as sf

# Load audio
waveform, sr = librosa.load("audio.wav", sr=22050, mono=True)
print(waveform.shape)   # (66150,)
print(sr)               # 22050
```

**Common sample rates:**

| Sample Rate | Used In |
|---|---|
| 8,000 Hz | Telephone, speech codecs |
| 16,000 Hz | Speech recognition (Whisper, Wav2Vec) |
| 22,050 Hz | Music analysis (librosa default) |
| 44,100 Hz | CD audio |
| 48,000 Hz | Professional audio / video |

---

## 2. Why Convert to Frequency Domain?

Models trained on raw waveforms must learn to detect frequencies from scratch — an extremely hard task given sequences of 100,000+ samples. The frequency domain:

- Compresses the signal dramatically (sequence length drops 100-200x)
- Makes frequency patterns visually and mathematically explicit
- Aligns with how the human auditory system works
- Converts 1D audio → 2D image → CNNs can be applied directly

---

## 3. Short-Time Fourier Transform (STFT)

The STFT slides a window across the waveform, computes the Fourier Transform inside each window, and stacks the results into a 2D matrix.

```
Window size (n_fft):    how many samples per FFT window
Hop length:             how many samples to advance between windows
Overlap:                n_fft - hop_length
```

```python
import librosa
import numpy as np

n_fft     = 2048   # FFT window size → frequency resolution
hop_length = 512   # step between windows → time resolution

# Compute STFT
stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
# stft shape: (n_fft//2 + 1, time_frames) = (1025, ~130) for 3s audio

# Convert complex STFT to magnitude (power) spectrogram
spectrogram = np.abs(stft) ** 2
print(spectrogram.shape)   # (1025, 130)
```

### STFT Parameters Trade-off

| Parameter | Effect |
|---|---|
| Large `n_fft` | Better frequency resolution, worse time resolution |
| Small `n_fft` | Better time resolution, worse frequency resolution |
| Small `hop_length` | Denser time axis (more overlap), larger output |
| Large `hop_length` | Coarser time axis, smaller output |

**Common settings:**

| Use Case | n_fft | hop_length |
|---|---|---|
| Speech recognition | 400–512 | 160 |
| Music analysis | 2048 | 512 |
| General audio | 1024 | 256 |

---

## 4. Window Functions

A window function is applied to each frame before FFT to reduce spectral leakage at frame boundaries.

```python
# Hann window (default and recommended for most audio)
stft = librosa.stft(waveform, n_fft=2048, hop_length=512, window="hann")

# Other options
# "hamming"  → slightly different roll-off
# "blackman" → better sidelobe suppression
# "boxcar"   → rectangular (no windowing) — not recommended
```

---

## 5. Power Spectrogram vs Magnitude Spectrogram

```python
stft = librosa.stft(waveform, n_fft=2048, hop_length=512)

# Magnitude spectrogram
mag_spec = np.abs(stft)                  # |STFT|

# Power spectrogram
power_spec = np.abs(stft) ** 2           # |STFT|²

# dB-scale spectrogram (most common for visualization and models)
db_spec = librosa.amplitude_to_db(mag_spec, ref=np.max)
```

---

## 6. Visualizing the Spectrogram

```python
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(12, 4))
librosa.display.specshow(
    db_spec,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="hz"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.tight_layout()
plt.show()
```

---

## 7. Output Shape Summary

```
Input waveform:   (N,)                      e.g. (66150,)
After STFT:       (n_fft//2 + 1, T)         e.g. (1025, 130)
After dB scale:   (n_fft//2 + 1, T)         same shape, log-compressed values
```

This 2D matrix is now an image-like structure:
- **Y-axis** → frequency bins (0 Hz to sr/2)
- **X-axis** → time frames
- **Value** → energy at that frequency at that time