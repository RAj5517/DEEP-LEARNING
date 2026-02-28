# Biology & Protein Transformers

> "A protein sequence is just another language. 20 amino acids. Billions of sequences. The transformer reads them all."

---

## The Core Insight

DNA and proteins are sequences. Amino acids are like tokens. The grammar is evolution.

A protein is a chain of amino acids (20 types, like letters of an alphabet) that folds into a 3D shape — and that shape determines its function. Predicting that shape from sequence alone was one of biology's hardest unsolved problems for 50 years.

Transformers solved it.

---

## Why Transformers Work for Biology

```
Text Language              vs        Protein Language
│                                    │
├── Tokens: words                    ├── Tokens: amino acids (20 types)
├── Grammar: syntax rules            ├── Grammar: evolutionary constraints
├── Meaning: semantics               ├── Meaning: 3D structure + function
└── Corpus: internet text            └── Corpus: UniProt (250M+ sequences)
```

Self-attention naturally captures **long-range dependencies** — just like a word at position 1 can influence meaning at position 500 in text, an amino acid at position 1 can form a contact with one at position 400 in a folded protein.

---

## Main Models

### AlphaFold2 (2021) — DeepMind
The model that solved protein structure prediction. Won the CASP14 competition (biennial competition for structure prediction) by an unprecedented margin — solving problems that had stumped researchers for decades.

Not a pure transformer — it's a hybrid:

```
AlphaFold2 Architecture
│
├── Input
│     ├── Amino acid sequence
│     └── Multiple Sequence Alignment (MSA) — evolutionary related sequences
│
├── Evoformer (48 layers)
│     ├── MSA Transformer       attention across evolutionary sequences
│     └── Pair Representation   attention over pairs of residues
│                               tracks which amino acids are close in 3D
│
└── Structure Module
      └── Iterative refinement → predicts 3D coordinates of every atom
```

AlphaFold2 predicted structures for nearly every known protein (~200 million) — the AlphaFold Protein Structure Database is now freely available.

**Impact**: Accelerated drug discovery, enzyme design, and fundamental biology research by years.
Paper: https://www.nature.com/articles/s41586-021-03819-2

---

### ESM — Evolutionary Scale Modeling (2023) — Meta AI
A family of protein language models trained purely on amino acid sequences — no structure, no MSA. Just sequence → representation.

```
ESM Model Family
│
├── ESM-1b (2020)      650M params · learned protein grammar from sequences
├── ESM-2  (2022)      up to 15B params · trained on 250M sequences
└── ESMFold (2022)     ESM-2 + structure head → fast structure prediction
                       10× faster than AlphaFold2 (no MSA needed)
```

ESMFold enables **high-throughput structure prediction** — predict millions of structures quickly.
Paper: https://www.science.org/doi/10.1126/science.ade2574

---

### ProtTrans (2021) — Rostlab
Applied standard NLP transformer architectures (BERT, GPT, T5, XLNet) directly to protein sequences with minimal modification. Trained on UniRef and BFD databases (393 billion amino acids — larger than all text used to train GPT-3 at the time).

Showed that standard transformers, scaled to biological sequence data, learn meaningful biological representations without structural supervision.
Paper: https://arxiv.org/abs/2007.06225

---

## Other Notable Models

| Model | By | Note |
|-------|----|------|
| ProtGPT2 | HuggingFace / Ferruz | GPT-2 trained on proteins → generate novel protein sequences |
| ProGen2 | Salesforce Research | Generate functional proteins with desired properties |
| RFdiffusion | Baker Lab (UW) | Diffusion model for protein design — design proteins from scratch |
| AlphaFold3 | DeepMind | Extended to DNA, RNA, small molecules — full biology |

---

## Use Cases

```
Biology Transformer Use Cases
│
├── Protein Structure Prediction    sequence → 3D structure (AlphaFold2)
├── Protein Design                  design novel proteins with desired properties
├── Drug Discovery                  find proteins that bind target drug molecules
├── Mutation Effect Prediction      will this mutation cause disease?
├── Enzyme Engineering              design enzymes for industrial applications
└── Genomics                        DNA sequence → gene function prediction
```

---

## Why This Matters

Before AlphaFold2: solving one protein structure took months to years in a lab.
After AlphaFold2: predict it in minutes with a computer.

The AlphaFold Protein Structure Database has structures for 200M+ proteins.
This is transforming drug discovery, disease research, and synthetic biology.

The 2024 Nobel Prize in Chemistry was awarded partly for AlphaFold2's contribution to protein structure prediction.

---

## Key Papers

| Paper | Link |
|-------|------|
| AlphaFold2 (2021) | https://www.nature.com/articles/s41586-021-03819-2 |
| ESM-2 / ESMFold (2022) | https://www.science.org/doi/10.1126/science.ade2574 |
| ProtTrans (2021) | https://arxiv.org/abs/2007.06225 |