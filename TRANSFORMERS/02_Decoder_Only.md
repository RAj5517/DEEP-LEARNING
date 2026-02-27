# Decoder-Only Transformers

> "Predict the next word. Repeat. That's how you write like a human."

---

## What It Does

A Decoder-Only model generates text **one token at a time**, from left to right. It looks at everything it has generated so far and predicts what comes next.

This is called **autoregressive generation**.

No encoder. No cross-attention. Just one stack of masked transformer layers that keep asking the same question: *given everything before this point, what word comes next?*

Scale this up to billions of parameters and trillions of tokens — and you get GPT-4, Claude, LLaMA.

---

## Architecture

```
Input (prompt tokens)
│
├── Token Embedding
├── Positional Encoding (or RoPE in modern models)
│
└── × N Decoder Layers
      ├── Masked Multi-Head Self-Attention
      │     └── Each token ONLY attends to past tokens — future is masked
      ├── Add & RMSNorm (modern) / LayerNorm (older)
      ├── Feed Forward Network (SwiGLU in modern models)
      └── Add & RMSNorm
│
└── Linear Layer → Vocab size (50,000+ tokens)
└── Softmax → Probability distribution
└── Sample / Argmax → Next token
└── Append → feed back as input → repeat
```

---

## Attention Type

**Masked (Causal) Self-Attention** — future tokens are hidden.

```
Generating: "The sky is blue"

When generating "blue":
  ✓ can see → "The", "sky", "is"
  ✗ cannot see → anything after "blue" (doesn't exist yet)
```

Masking is done by setting future positions to -∞ before softmax.
This forces the model to never "cheat" by looking ahead.

---

## Pre-Training Objective

**Next Token Prediction (Causal Language Modeling)**

Given a sequence, predict the next token at every position.

```
Input:   "The cat sat on the"
Target:  "cat sat on the mat"
```

Simple objective. Massive scale. Surprisingly powerful.

This is called **emergent behavior** — at large enough scale, the model develops abilities nobody explicitly trained for: reasoning, coding, math, translation.

---

## Modern Improvements over Original GPT

```
Modern Decoder Improvements
│
├── Positional Encoding
│     ├── Original          Absolute sinusoidal (GPT-1, GPT-2)
│     └── Modern            RoPE — Rotary Position Embedding (LLaMA, Mistral)
│                           better length generalization
│
├── Normalization
│     ├── Original          Post-LayerNorm (BERT, GPT-2)
│     └── Modern            Pre-RMSNorm — more stable training (LLaMA)
│
├── Activation
│     ├── Original          ReLU / GeLU
│     └── Modern            SwiGLU — smoother, better performance
│
├── Attention Efficiency
│     ├── GQA               Grouped Query Attention — fewer KV heads (Mistral)
│     ├── MQA               Multi Query Attention — single KV head
│     └── FlashAttention    IO-aware exact attention — faster on GPU
│
└── Context Length
      ├── GPT-2             1,024 tokens
      ├── GPT-3             4,096 tokens
      └── Modern LLMs       128K → 1M+ tokens (Gemini 1.5)
```

---

## Use Cases

```
Decoder-Only Use Cases
│
├── Chat / Conversational AI     ChatGPT · Claude · Gemini
├── Code Generation              Copilot · Cursor · CodeLLaMA
├── Text Completion              autocomplete · creative writing
├── Reasoning                    chain-of-thought · math · logic
├── Instruction Following        "explain X" · "write Y" · "fix Z"
└── Few-Shot Learning            examples in prompt → new task without training
```

---

## Main Models

### GPT-3 (2020) — OpenAI
The model that showed the world what scale can do.
175 billion parameters. No fine-tuning needed — just describe the task in the prompt (few-shot learning). Could write essays, answer questions, generate code — without being explicitly trained for any of it.

- First model to exhibit strong emergent behavior
- Introduced the idea of prompting as programming
- Paper: https://arxiv.org/abs/2005.14165

---

### GPT-4 (2023) — OpenAI
Multimodal. Significantly more capable than GPT-3. Architecture details not fully disclosed (rumored MoE). Powers ChatGPT. Excels at reasoning, coding, instruction following.
Paper: https://arxiv.org/abs/2303.08774

---

### LLaMA 2 / LLaMA 3 (2023/2024) — Meta
Open-source. Changed everything — anyone could now run a powerful LLM locally. Uses RoPE, RMSNorm, SwiGLU, GQA. Trained on 2T tokens. Foundation for most open-source fine-tunes (Alpaca, Vicuna, WizardLM).

- LLaMA 3 (8B) matches GPT-3.5 on many benchmarks
- Paper: https://arxiv.org/abs/2302.13971

---

### Mistral 7B (2023) — Mistral AI
7 billion parameters but punches far above its weight. Uses:
- **Sliding Window Attention** — each token attends to only 4096 previous tokens (efficient)
- **Grouped Query Attention (GQA)** — faster inference
- Outperforms LLaMA 2 13B despite being smaller

Paper: https://arxiv.org/abs/2310.06825

---

### Claude (Anthropic)
Decoder-only transformer trained with RLHF and Constitutional AI. Emphasis on safety, helpfulness, and long-context understanding. Claude 3 models (Haiku, Sonnet, Opus) range from fast/cheap to highly capable.

---

### Other Notable Models

| Model | By | Key Note |
|-------|----|----------|
| Falcon | TII UAE | Strong open-source · trained on RefinedWeb |
| Gemini | Google DeepMind | Natively multimodal from day one |
| Phi-2 / Phi-3 | Microsoft | Tiny but powerful — trained on high-quality data |
| DeepSeek | DeepSeek AI | Strong at coding and math · open weights |
| Mixtral | Mistral AI | Mixture of Experts — 8 experts · 2 active per token |

---

## RLHF — How Raw LLMs Become Assistants

Pre-trained decoder models predict next tokens well but don't follow instructions.
RLHF (Reinforcement Learning from Human Feedback) fixes this:

```
Step 1 — Supervised Fine-Tuning (SFT)
  Train on human-written prompt-response pairs

Step 2 — Reward Model Training
  Humans rank multiple model responses
  Train a reward model to predict human preference

Step 3 — PPO Reinforcement Learning
  Fine-tune the LLM using the reward model as signal
  Model learns to generate responses humans prefer
```

This turned GPT-3 into InstructGPT → ChatGPT.
Paper: https://arxiv.org/abs/2203.02155

---

## Key Papers

| Paper | Link |
|-------|------|
| GPT-1 (2018) | https://openai.com/research/language-unsupervised |
| GPT-3 (2020) | https://arxiv.org/abs/2005.14165 |
| InstructGPT / RLHF (2022) | https://arxiv.org/abs/2203.02155 |
| LLaMA (2023) | https://arxiv.org/abs/2302.13971 |
| Mistral 7B (2023) | https://arxiv.org/abs/2310.06825 |