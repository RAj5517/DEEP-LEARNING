# Multimodal Transformers

> "Language is just one way to understand the world. What if the model could see, hear, and read — all at once?"

---

## What It Does

Multimodal Transformers process **more than one type of input** — typically text + images, but also audio, video, and structured data.

The key challenge: text is a sequence of tokens. Images are grids of pixels. Audio is a waveform. These are completely different modalities with different structures. The Transformer has to learn a **shared representation space** where all of them can interact.

---

## Core Approaches

```
How Multimodal Models Handle Different Inputs
│
├── Dual Encoder (two separate encoders → align in shared space)
│     └── CLIP — encode image and text separately · train to match
│
├── Fusion Model (combine modalities inside transformer layers)
│     ├── Early Fusion     concatenate tokens from all modalities at input
│     └── Late Fusion      encode separately · merge at higher layers
│
└── Generative Multimodal (single model that reads and generates multiple types)
      └── GPT-4V · Gemini · LLaVA — image tokens + text tokens → generation
```

---

## Architecture Patterns

### Pattern 1 — Contrastive (CLIP-style)

```
Image → ViT Encoder → Image embedding [512-dim]
                                         ↓
                                   Cosine Similarity → contrastive loss
                                         ↑
Text  → Text Encoder → Text embedding  [512-dim]
```

Train on 400M image-text pairs scraped from the internet.
Goal: matching image and text embeddings should be similar, non-matching should be far apart.

---

### Pattern 2 — Generative (LLaVA / GPT-4V style)

```
Image
  ↓
Vision Encoder (ViT)
  ↓
Visual Projection Layer     → map visual features to LLM embedding space
  ↓
[Visual tokens] + [Text tokens]
  ↓
Large Language Model (decoder-only)
  ↓
Generated text response
```

The LLM doesn't see pixels — it sees visual features projected into the same space as word embeddings. Then it attends to both visual and text tokens together.

---

## Use Cases

```
Multimodal Use Cases
│
├── Visual Question Answering    "What color is the car in this image?"
├── Image Captioning             image → describe what you see
├── Document Understanding       PDF · chart · table → answer questions
├── Text-to-Image Generation     "a cat on a spaceship" → image
├── Video Understanding          watch video → summarize · answer questions
├── Medical Image + Report       X-ray + clinical notes → diagnosis
└── OCR + Language               image of text → extract + understand
```

---

## Main Models

### CLIP — Contrastive Language-Image Pretraining (2021) — OpenAI
The model that made zero-shot image classification mainstream.

Trained on 400M image-text pairs. Learns a shared embedding space where images and their descriptions end up close together.

**Zero-shot classification**: describe classes in text ("a photo of a dog", "a photo of a cat"), encode them, encode the query image — find which text is closest. No training needed for new classes.

Used as the visual backbone in DALL-E 2, Stable Diffusion, and many other systems.
Paper: https://arxiv.org/abs/2103.00020

---

### DALL-E 2 (2022) — OpenAI
Text → Image generation. Uses CLIP embeddings as the bridge between language and image space. Takes text, encodes it with CLIP text encoder, then diffusion model generates an image whose CLIP embedding matches.

Architecture: CLIP + Diffusion Model (not pure transformer, but CLIP component is)
Paper: https://arxiv.org/abs/2204.06125

---

### GPT-4V (2023) — OpenAI
GPT-4 with vision. Images are tokenized and fed alongside text into the LLM's context window. The model can reason about images, read text in images, interpret charts, understand diagrams, and generate text responses about visual content.

Capability highlights:
- Read and understand handwritten notes
- Interpret medical scans with description
- Debug code from a screenshot
- Understand memes (context + image)

Paper: https://arxiv.org/abs/2303.08774

---

### Gemini (2023) — Google DeepMind
Built natively multimodal from day one — trained on text, images, audio, and video simultaneously. Not text-first with vision bolted on.

Three sizes: Gemini Nano (on-device) · Gemini Pro · Gemini Ultra
Gemini 1.5 extended context to 1M tokens — can process entire movies or codebases.

Paper: https://arxiv.org/abs/2312.11805

---

### LLaVA — Large Language and Vision Assistant (2023)
Open-source. Simple and powerful approach:
1. Encode image with CLIP ViT
2. Project visual features to LLM embedding space with a linear layer
3. Concatenate with text tokens
4. Feed into LLaMA / Mistral

Trained on GPT-4 generated image description data. Surprisingly capable for its simplicity. Foundation of many open-source vision-language models.

Paper: https://arxiv.org/abs/2304.08485

---

### Flamingo (2022) — DeepMind
**Few-shot visual language model.** Can handle interleaved sequences of images and text. Takes a sequence like: `[image1] [text] [image2] [text] → generate answer`.

Key innovation: **Perceiver Resampler** — compresses variable-size visual features into a fixed number of visual tokens, then injects them into frozen LLM layers via cross-attention.

Paper: https://arxiv.org/abs/2204.14198

---

## CLIP — Deep Dive (Most Widely Used)

```
Training Objective (Contrastive)
│
├── Batch of N image-text pairs
├── Encode all N images    → N image embeddings
├── Encode all N texts     → N text embeddings
│
└── Maximize similarity of N correct pairs
    Minimize similarity of N² - N incorrect pairs
```

After training, CLIP understands concepts like:
- "a dog running in a park"
- "a photo of a cat"
- "satellite image of a city at night"
- "a chest X-ray showing pneumonia"

Without ever being fine-tuned on those specific tasks.

---

## Key Papers

| Paper | Link |
|-------|------|
| CLIP (2021) | https://arxiv.org/abs/2103.00020 |
| Flamingo (2022) | https://arxiv.org/abs/2204.14198 |
| DALL-E 2 (2022) | https://arxiv.org/abs/2204.06125 |
| GPT-4 / GPT-4V (2023) | https://arxiv.org/abs/2303.08774 |
| LLaVA (2023) | https://arxiv.org/abs/2304.08485 |
| Gemini (2023) | https://arxiv.org/abs/2312.11805 |