# 🎵 Audio Processing — Preprocessing Pipeline

> Models don't hear sound.
> They read 2D grids of frequency energy over time.
> Everything between a raw audio file and model input is **audio preprocessing**.

---

## 🔁 Pipeline at a Glance

```
Raw Waveform          (amplitude samples over time)
   ↓
Waveform Preprocessing   (resample · normalize · pad)
   ↓
Spectrogram              (STFT → time-frequency representation)
   ↓
Mel-Spectrogram          (human hearing aligned scaling)
   ↓
Normalization            (dB compression · Z-score · SpecAugment)
   ↓
Tensor Conversion        (add channel dim → GPU)
   ↓
Model  (Speech / Music / Voice AI)
```

---

## 📂 Structure

| File | Covers |
|---|---|
| `01_spectrogram.md` | STFT, window functions, n_fft vs hop_length trade-off |
| `02_mel_spectrogram.md` | Mel scale, filter bank, MFCC, torchaudio implementation |
| `03_audio_normalization.md` | dB conversion, Z-score, SpecAugment, waveform augmentation |

---

## ❓ Why This Matters

| Problem | Caused By |
|---|---|
| Model sees 100,000+ raw samples | No STFT — waveform too long to model directly |
| High frequencies dominate | Linear spectrogram — no Mel scaling |
| Numerically unstable training | No dB conversion — power values span 6+ orders of magnitude |
| Overfitting on speaker identity | No SpecAugment — model memorizes audio artifacts |
| Variable-length crash | No padding/truncation — sequences can't batch |

---

## ⚡ When to Apply Each Step

| Step | Training | Validation | Inference |
|---|---|---|---|
| Resample + Mono | ✅ | ✅ | ✅ |
| Peak Normalize | ✅ | ✅ | ✅ |
| STFT → Mel-Spec | ✅ | ✅ | ✅ |
| dB Conversion | ✅ | ✅ | ✅ |
| Z-score Normalize | ✅ | ✅ | ✅ |
| SpecAugment | ✅ | ❌ | ❌ |

---

## 🔬 Core Idea

Sound is waves → STFT converts waves to frequencies → Mel scale aligns to human perception → log compression stabilizes values → model reads it like an image.

**A Mel-Spectrogram is a photograph of sound.**

---

*For deep breakdowns, math, and code — refer to the individual files above.*

┌──────────────────────────────────────────┐
  │     RAW WAVEFORM  (amplitude over time)  │
  └──────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │       1. WAVEFORM PREPROCESSING          │
  └──────────────────────────────────────────┘
  │
  ├── Resample            →  uniform sample rate (16000 / 22050 Hz)
  ├── Convert to Mono     →  stereo → single channel
  ├── Peak Normalize      →  waveform / max  →  [-1.0, 1.0]
  └── Pad / Truncate      →  fixed length for batching
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │     2. SPECTROGRAM  (Time → Frequency)   │
  └──────────────────────────────────────────┘
  │
  ├── STFT  (Short-Time Fourier Transform)
  │     ├── n_fft       →  frequency resolution
  │     ├── hop_length  →  time resolution
  │     └── window      →  Hann (default)
  │
  └── Output  (n_fft//2 + 1,  T)  →  e.g. (1025, 130)
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │        3. MEL-SPECTROGRAM                │
  └──────────────────────────────────────────┘
  │
  ├── Apply Mel Filter Bank    →  human hearing aligned scaling
  │     n_mels = 80            →  Whisper / speech recognition
  │     n_mels = 128           →  music analysis
  │
  ├── Log Compression (dB)     →  power_to_db()
  │     Raw power range  →  [0, 1e6+]
  │     After dB scale   →  [-80, 0]
  │
  ├── MFCC  (optional, traditional NLP)
  │     Mel-Spec → DCT → keep first 13–40 coefficients
  │
  └── Output  (n_mels, T)  →  e.g. (128, 130)
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │          4. NORMALIZATION                │
  └──────────────────────────────────────────┘
  │
  ├── Z-score  (per sample)
  │     (x − mean) / std   →  zero-centered
  │
  ├── Z-score  (dataset-level)
  │     global mean & std across all training files
  │
  └── SpecAugment  (training only)
        ├── Frequency Masking  →  mask F consecutive frequency bins
        └── Time Masking       →  mask T consecutive time steps
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │          5. TENSOR CONVERSION            │
  └──────────────────────────────────────────┘
  │
  ├── Add channel dim    →  (n_mels, T)  →  (1, n_mels, T)
  └── Move to GPU        →  tensor.to(device)
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │    MODEL  (Speech / Music / Voice AI)    │
  │    CNN · RNN · Transformer · Whisper     │
  └──────────────────────────────────────────┘