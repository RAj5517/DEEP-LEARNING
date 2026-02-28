# Audio & Speech Transformers

> "Sound is a wave. Slice it into frames. Each frame becomes a token. Now the transformer can listen."

---

## How Audio Becomes Tokens

Unlike text (discrete words) or images (2D pixel grids), audio is a **continuous 1D waveform** — millions of samples per second.

Transformers need sequences of tokens. The solution:

```
Raw Audio Waveform
│
└── Short-Time Fourier Transform (STFT)
      ↓
Log-Mel Spectrogram
  → 2D image-like representation
  → X-axis: time frames (~10ms each)
  → Y-axis: frequency bins (mel scale)
      ↓
Slice into time frames OR patches
      ↓
Each frame/patch = one token → feed into transformer
```

A mel spectrogram turns sound into something that looks like an image — and we can apply transformers (or even ViT-style patch tokenization) to it.

---

## Main Models

### Whisper (2022) — OpenAI
The most widely used speech recognition model today. An Encoder-Decoder transformer trained on **680,000 hours of multilingual audio** scraped from the internet — 100× more data than previous models.

```
Whisper Architecture
│
├── Audio Input → Log-Mel Spectrogram (80-channel)
├── CNN stem → extract local features first
│
├── ENCODER (Transformer)
│     └── Attend across time frames — understand the audio
│
└── DECODER (Transformer)
      └── Generate text transcript token by token
          (or translated text for translation task)
```

Capabilities:
- Speech-to-text in 99 languages
- Language identification
- Timestamp generation
- Translation to English from any of 99 languages

The key insight: massive diverse data made it robust. Works on accents, background noise, technical jargon, and rare languages that previous models failed on.

Sizes: Tiny (39M) · Base (74M) · Small (244M) · Medium (769M) · Large (1.5B)
Paper: https://arxiv.org/abs/2212.04356

---

### Wav2Vec 2.0 (2020) — Facebook AI
**Self-supervised speech representation learning** — learns from raw audio without transcription labels.

```
Wav2Vec 2.0 Training
│
├── Raw audio waveform
├── CNN feature extractor → local audio features (25ms frames)
├── Quantization module → discretize into speech units (like a "sound vocabulary")
│
└── Transformer encoder (BERT-style)
      ├── Mask random spans of audio frames
      └── Predict the quantized representation of masked frames
          (like MLM but for sound)
```

Result: learns rich speech representations using unlabeled audio. Fine-tune with just 10 minutes of labeled data to get competitive speech recognition.

This matters enormously for low-resource languages where transcribed data is scarce.
Paper: https://arxiv.org/abs/2006.11477

---

### HuBERT — Hidden-Unit BERT (2021) — Facebook AI
Improved on Wav2Vec 2.0. Instead of quantizing audio during training, it uses offline clustering (k-means) to create pseudo-labels, then trains BERT-style on those labels.

Iterative process: cluster → train → re-cluster with better representations → re-train.
Achieved state-of-the-art on LibriSpeech benchmark at time of release.
Paper: https://arxiv.org/abs/2106.07447

---

### AudioLM (2022) — Google
Goes beyond speech recognition — generates audio.

```
AudioLM — Hierarchical Audio Generation
│
├── Semantic tokens (HuBERT)      high-level content and meaning
└── Acoustic tokens (SoundStream) low-level sound details · speaker voice
│
└── Transformer generates semantic tokens first
    → then generates acoustic tokens conditioned on semantic
    → decode back to audio waveform
```

Can continue any audio clip naturally — if you give it 3 seconds of someone speaking, it continues speaking in the same voice, with coherent content. Extends to music continuation.
Paper: https://arxiv.org/abs/2209.03143

---

## Other Notable Models

| Model | By | Note |
|-------|----|------|
| SpeechT5 | Microsoft | Unified encoder-decoder for speech tasks (ASR, TTS, VC) |
| VALL-E | Microsoft | TTS model that clones voice from 3s sample |
| MusicGen | Meta | Text-conditioned music generation transformer |
| AudioCraft | Meta | Suite of audio generation models (music + sound effects) |
| Seamless | Meta | Real-time speech translation across 100+ languages |

---

## Use Cases

```
Audio Transformer Use Cases
│
├── Automatic Speech Recognition (ASR)   speech → text (Whisper)
├── Text-to-Speech (TTS)                 text → human-like voice
├── Voice Cloning                        clone any voice from short sample
├── Speaker Identification               who is speaking?
├── Emotion Recognition                  detect emotion from voice tone
├── Music Generation                     generate music from text description
├── Sound Classification                 what sound is this? (gunshot · rain · speech)
└── Real-time Translation                speak English → hear French in real time
```

---

## Key Papers

| Paper | Link |
|-------|------|
| Wav2Vec 2.0 (2020) | https://arxiv.org/abs/2006.11477 |
| HuBERT (2021) | https://arxiv.org/abs/2106.07447 |
| AudioLM (2022) | https://arxiv.org/abs/2209.03143 |
| Whisper (2022) | https://arxiv.org/abs/2212.04356 |