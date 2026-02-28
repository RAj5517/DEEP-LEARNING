# Code Transformers

> "Code is just another language. Tokens, sequences, patterns — transformers read it the same way."

---

## Why Code is Different from Text

Natural language is ambiguous. Code is not.
- Syntax must be exact
- Variable names carry meaning across hundreds of lines
- Indentation, brackets, semicolons matter
- The model needs to understand **program flow**, not just word sequences

Code models are pre-trained on massive repositories of code — GitHub, StackOverflow, documentation — and learn both syntax and semantics of programming languages.

---

## Architecture

Most code models are **Decoder-Only** (like GPT) — they generate code token by token.
Some are **Encoder-Only** — used for code understanding, bug detection, search.

```
Code Model Architecture (Decoder-Only)
│
├── Tokenizer         BPE tokenizer trained on code
│     └── Understands identifiers · operators · indentation as tokens
│
├── Same decoder-only architecture as LLMs
│     └── Masked self-attention · FFN · RMSNorm
│
├── Pre-training data
│     ├── GitHub repositories (Python, Java, C++, JS, ...)
│     ├── StackOverflow Q&A
│     ├── Documentation and tutorials
│     └── Natural language descriptions paired with code
│
└── Special capabilities learned
      ├── Fill-in-the-Middle (FIM)   complete code given prefix + suffix
      ├── Docstring → code           generate implementation from description
      └── Code → docstring           generate documentation from code
```

---

## Use Cases

```
Code Transformer Use Cases
│
├── Code Completion          autocomplete current line / block
├── Code Generation          describe task in English → get working code
├── Bug Detection            find errors · suggest fixes
├── Code Explanation         what does this function do?
├── Refactoring              improve structure · rename · optimize
├── Test Generation          write unit tests for this function
├── Code Translation         Python → JavaScript · Java → Kotlin
└── Semantic Code Search     find function that does X (not exact match)
```

---

## Main Models

### Codex (2021) — OpenAI
Fine-tuned GPT-3 on 54 million GitHub repositories. Powered the original GitHub Copilot. Could solve 28.8% of HumanEval benchmark problems (Python coding problems) — GPT-3 solved 0%.

Showed that scale + code data = surprisingly capable programming assistant.
Paper: https://arxiv.org/abs/2107.03374

---

### CodeLLaMA (2023) — Meta
Built on LLaMA 2. Three variants:

```
CodeLLaMA variants
│
├── CodeLLaMA Base          pretrained on 500B code tokens
├── CodeLLaMA-Instruct      fine-tuned to follow instructions in natural language
└── CodeLLaMA-Python        further fine-tuned specifically on Python
```

Sizes: 7B · 13B · 34B. Context length up to 100K tokens — can handle large codebases.
Supports Fill-in-the-Middle: given prefix + suffix, complete the middle.

Paper: https://arxiv.org/abs/2308.12950

---

### StarCoder (2023) — BigCode (HuggingFace + ServiceNow)
Trained on The Stack — a 6.4TB dataset of permissively licensed code from GitHub across 80+ programming languages.

Key features:
- 8K context length
- Fill-in-the-Middle capability
- Multi-query attention for faster inference
- Fully open — model weights + training data available

StarCoder 2 (2024) extended to 16K context, trained on 3.3T tokens.
Paper: https://arxiv.org/abs/2305.06161

---

### DeepSeek-Coder (2024) — DeepSeek AI
Strong open-source code model. Trained on 2T tokens of code and natural language.

```
DeepSeek-Coder highlights
│
├── Sizes              1.3B · 6.7B · 33B
├── Context            16K tokens
├── Languages          87 programming languages
├── Benchmark          Outperforms GPT-3.5 on HumanEval at 33B size
└── Open weights       yes — Apache 2.0
```

Paper: https://arxiv.org/abs/2401.14196

---

## Other Notable Models

| Model | By | Note |
|-------|----|------|
| GitHub Copilot | OpenAI / GitHub | Product built on Codex → now GPT-4 based |
| WizardCoder | WizardLM | Fine-tuned StarCoder with evolved instructions |
| Phind-CodeLLaMA | Phind | Strong at complex problem solving |
| Qwen-Coder | Alibaba | Strong multilingual code model |

---

## Fill-in-the-Middle (FIM) — Key Technique

Standard generation: given prefix → generate suffix.
FIM: given prefix + suffix → generate middle.

```
Prefix:   def calculate_area(radius):
              """Calculate area of circle"""

Suffix:       return area

Middle:   [MODEL FILLS IN]  →  area = 3.14159 * radius ** 2
```

This is essential for IDE autocomplete — the cursor is in the middle of existing code.

---

## Key Papers

| Paper | Link |
|-------|------|
| Codex (2021) | https://arxiv.org/abs/2107.03374 |
| StarCoder (2023) | https://arxiv.org/abs/2305.06161 |
| CodeLLaMA (2023) | https://arxiv.org/abs/2308.12950 |
| DeepSeek-Coder (2024) | https://arxiv.org/abs/2401.14196 |