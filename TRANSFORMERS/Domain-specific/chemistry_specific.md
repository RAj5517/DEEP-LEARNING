# Chemistry Transformers

> "A molecule is a graph. An atom is a token. The bonds between them are relationships — and attention learns them."

---

## The Core Idea

Molecules can be represented as:
- **SMILES strings** — a text notation: `CC(=O)OC1=CC=CC=C1C(=O)O` = Aspirin
- **Graphs** — atoms as nodes, bonds as edges
- **3D coordinates** — atom positions in space

Transformers work beautifully on SMILES strings (treat them as text sequences) and can also be adapted to molecular graphs.

---

## Why Chemistry Needs Transformers

Traditional drug discovery: synthesize thousands of compounds in a lab. Test each one. Takes years. Costs billions.

ML-powered discovery: train a model on millions of known compounds and their properties. Use it to predict which new molecules might be drug candidates — before synthesizing a single one.

Transformers learn the implicit "grammar" of chemistry — which atoms bond with which, what makes a molecule stable, which functional groups predict certain properties.

---

## Molecular Representation

```
Representing a Molecule for a Transformer
│
├── SMILES string (text-based)
│     └── "CC(=O)Oc1ccccc1C(=O)O"  →  tokenize like text → transformer
│
├── SELFIES (more robust than SMILES)
│     └── always valid molecular strings — better for generation
│
└── Graph-based
      ├── Atoms → nodes (token-like)
      ├── Bonds → edges (relationships)
      └── Graph Transformer — attention over graph structure
```

---

## Main Models

### ChemBERTa (2020) — Chithrananda et al.
BERT applied directly to SMILES strings. Pre-trained on 77M molecules from PubChem using masked language modeling — randomly mask atoms in a SMILES string and predict them.

Learns representations of molecules that can be fine-tuned for:
- Predicting solubility
- Predicting toxicity
- Predicting binding affinity

Shows that the standard BERT approach transfers directly to chemistry with minimal modification.
Paper: https://arxiv.org/abs/2010.09885

---

### MolBERT (2020) — Fabian et al.
Similar to ChemBERTa but with additional pre-training objectives:
- SMILES equivalence — two different SMILES for the same molecule should have similar embeddings
- Physicochemical property prediction during pre-training

Better molecular representations for downstream property prediction tasks.
Paper: https://arxiv.org/abs/2011.13230

---

## Other Notable Models

| Model | By | Note |
|-------|----|------|
| Mol2Vec | Jaeger et al. | Word2Vec for molecules — embeddings for substructures |
| GROVER | Tencent | Graph Transformer pre-trained on 10M molecules |
| MegaMolBART | NVIDIA | BART for molecule generation and optimization |
| Chemformer | AstraZeneca | Encoder-decoder for reaction prediction and molecule generation |
| AlphaFold3 | DeepMind | Handles small molecules + proteins together |

---

## Use Cases

```
Chemistry Transformer Use Cases
│
├── Molecular Property Prediction    solubility · toxicity · bioactivity
├── Drug-Target Interaction          will this molecule bind this protein?
├── De Novo Drug Design              generate novel molecules with desired properties
├── Reaction Prediction              given reactants → predict products
├── Retrosynthesis                   given target molecule → what reactions produce it?
└── ADMET Prediction                 Absorption · Distribution · Metabolism · Excretion · Toxicity
```

---

## The Big Picture

```
Traditional Drug Discovery (decades)
  Hypothesis → Synthesize → Test → Fail → Repeat

ML-Accelerated Discovery
  Train on known molecules → Generate candidates → Filter by predicted properties
  → Synthesize only the best candidates → Test
```

Companies like Insilico Medicine, Recursion Pharmaceuticals, and Schrödinger use transformer-based models as core tools in their drug discovery pipelines. AI-designed molecules have already entered human clinical trials.

---

## Key Papers

| Paper | Link |
|-------|------|
| ChemBERTa (2020) | https://arxiv.org/abs/2010.09885 |
| MolBERT (2020) | https://arxiv.org/abs/2011.13230 |
| GROVER (2020) | https://arxiv.org/abs/2007.02835 |