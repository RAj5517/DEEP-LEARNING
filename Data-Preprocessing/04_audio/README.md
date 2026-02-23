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
