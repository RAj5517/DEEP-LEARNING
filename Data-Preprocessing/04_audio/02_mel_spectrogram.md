# Mel-Spectrogram — Human Hearing Aligned Scaling

A regular spectrogram treats all frequencies equally — 100 Hz to 200 Hz gets the same space as 10,000 Hz to 10,100 Hz. But humans don't hear that way. We are far more sensitive to differences in low frequencies than high frequencies. The Mel scale warps the frequency axis to match human auditory perception.

---

## 1. The Mel Scale

The Mel scale is a perceptual scale of pitch. It compresses high frequencies and expands low frequencies, mirroring how the human cochlea processes sound.

```
Mel(f) = 2595 × log10(1 + f / 700)

Examples:
  100 Hz  →  150 Mel
  1000 Hz →  999 Mel
  4000 Hz →  1920 Mel
  8000 Hz →  2840 Mel
```

The key insight: the difference between 100 Hz and 200 Hz sounds much larger to us than the difference between 5000 Hz and 5100 Hz — even though both are 100 Hz apart. The Mel scale captures this.

---

## 2. Mel Filter Bank

A Mel filter bank is a set of triangular filters placed on the frequency axis at Mel-spaced intervals. Each filter sums the energy in its frequency range into a single value.

```
n_mels = 80 filters → each covers a Mel-spaced frequency band
Output: 80 values per time frame instead of 1025
```

This reduces the frequency dimension from `n_fft//2 + 1` (e.g. 1025) down to `n_mels` (typically 64–128), while preserving perceptually relevant structure.

---

## 3. Computing Mel-Spectrogram

```python
import librosa
import numpy as np

waveform, sr = librosa.load("audio.wav", sr=22050)

mel_spec = librosa.feature.melspectrogram(
    y          = waveform,
    sr         = sr,
    n_fft      = 2048,      # FFT window size
    hop_length = 512,       # step between windows
    n_mels     = 128,       # number of Mel filter bands
    fmin       = 0,         # minimum frequency (Hz)
    fmax       = 8000       # maximum frequency (Hz) — sr/2 if None
)

print(mel_spec.shape)   # (128, 130)  →  (n_mels, time_frames)

# Convert to dB scale — critical for numerical stability in models
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
print(mel_db.shape)     # (128, 130)
print(mel_db.min(), mel_db.max())   # ~(-80, 0) dB
```

---

## 4. n_mels — Choosing the Right Value

| n_mels | Use Case |
|---|---|
| 40 | Lightweight speech models, MFCC extraction |
| 64 | General speech processing |
| 80 | Whisper (OpenAI), speech recognition standard |
| 128 | Music analysis, high-resolution audio |
| 229 | High-fidelity music generation |

---

## 5. MFCC — Mel Frequency Cepstral Coefficients

MFCCs are a further compression of the Mel-Spectrogram. A Discrete Cosine Transform (DCT) is applied to the log Mel energies, producing coefficients that decorrelate the features and capture the overall spectral shape.

```
Log Mel-Spectrogram → DCT → MFCCs

Steps:
1. Compute Mel-Spectrogram
2. Take log (log compression)
3. Apply DCT → keep first N coefficients
```

```python
mfcc = librosa.feature.mfcc(
    y       = waveform,
    sr      = sr,
    n_mfcc  = 13,     # number of MFCC coefficients (typically 13–40)
    n_mels  = 128,
    n_fft   = 2048,
    hop_length = 512
)
print(mfcc.shape)   # (13, 130)

# Delta features — capture rate of change over time
mfcc_delta  = librosa.feature.delta(mfcc)          # 1st derivative
mfcc_delta2 = librosa.feature.delta(mfcc, order=2) # 2nd derivative

# Stack all three → (39, T)
mfcc_full = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
```

**MFCCs vs Mel-Spectrogram:**

| | MFCC | Mel-Spectrogram |
|---|---|---|
| Dimensionality | Very low (13–40) | Medium (64–128) |
| Information | Spectral shape only | Full spectral detail |
| Traditional NLP / HMM | ✅ Preferred | ❌ |
| Deep learning (CNN/Transformer) | ❌ | ✅ Preferred |

---

## 6. Other Spectral Features

```python
# Spectral centroid — "brightness" of the sound (weighted mean frequency)
centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)

# Spectral bandwidth — width of the frequency spread
bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)

# Spectral rolloff — frequency below which X% of energy is contained
rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)

# Zero crossing rate — how often signal crosses zero (percussiveness indicator)
zcr = librosa.feature.zero_crossing_rate(waveform)

# Chroma features — pitch class energy (musical key detection)
chroma = librosa.feature.chroma_stft(y=waveform, sr=sr, n_chroma=12)
```

---

## 7. Using PyTorch torchaudio

```python
import torchaudio
import torchaudio.transforms as T

waveform, sr = torchaudio.load("audio.wav")

# Resample if needed
resampler = T.Resample(orig_freq=sr, new_freq=16000)
waveform = resampler(waveform)

# Mel-Spectrogram
mel_transform = T.MelSpectrogram(
    sample_rate = 16000,
    n_fft       = 400,
    hop_length  = 160,
    n_mels      = 80,
    f_min       = 0,
    f_max       = 8000
)
mel_spec = mel_transform(waveform)   # shape: (1, 80, T)

# Convert to dB
db_transform = T.AmplitudeToDB()
mel_db = db_transform(mel_spec)
print(mel_db.shape)   # (1, 80, T)
```

---

## 8. Shape Summary

```
Raw waveform:       (N,)                    e.g. (66150,)
STFT magnitude:     (n_fft//2 + 1, T)       e.g. (1025, 130)
Mel-Spectrogram:    (n_mels, T)             e.g. (128, 130)
Log Mel (dB):       (n_mels, T)             e.g. (128, 130)  values ~[-80, 0]
MFCC:               (n_mfcc, T)             e.g. (13, 130)

→ Add channel dim for CNN:  (1, n_mels, T)  e.g. (1, 128, 130)
```