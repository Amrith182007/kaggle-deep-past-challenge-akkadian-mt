# 🏺 Deep Past Akkadian MT

> Low-resource **Akkadian → English Neural Machine Translation** using **NLLB-200**, **LoRA fine-tuning**, and **data-centric preprocessing** for the Kaggle Deep Past Challenge.

---

## 📖 Overview

This project explores neural machine translation for one of the most challenging low-resource language tasks: translating **4,000-year-old Old Assyrian Akkadian cuneiform transliterations** into English.

Built for the **Kaggle Deep Past Initiative Challenge**, the system fine-tunes Meta's **NLLB-200** models using:
- sentence-level corpus reconstruction
- Akkadian text normalization
- constrained decoding
- TPU-based LoRA fine-tuning
- custom preprocessing pipelines

The project focuses heavily on **data-centric NLP engineering**, tokenizer diagnostics, and low-resource MT robustness.

---

## ✨ Key Features

- 🏛️ **Ancient Language MT**  
  Translation of Akkadian cuneiform transliterations (1950–1750 BCE) into English.

- 📊 **Data-Centric Pipeline**  
  Expanded training data from ~1,561 document-level pairs to **5,700+ sentence-level pairs** using corpus reconstruction and OARE alignment.

- 🔧 **LoRA Fine-Tuning**  
  Parameter-efficient adaptation (`r=32`) on both **NLLB-200 600M** and **1.3B** models.

- 🐛 **Critical Tokenizer Bug Fix**  
  Diagnosed and fixed a silent `<gap>` token decoding issue worth **+1.74 leaderboard points**.

- 🔍 **Offline-Compatible Group Beam Search**  
  Pinned `transformers==4.38.2` to preserve native Group Beam Search support for Kaggle offline inference.

- ⚡ **Custom TPU v5e-8 Training Loop**  
  Replaced HuggingFace Trainer with a custom PyTorch + XLA loop to avoid TPU deadlocks.

- 📝 **Constrained Decoding**  
  Implemented lexicon-based proper noun forcing with N-best reranking.

---

## 📈 Results

| Model | Public LB | Private LB | Val BLEU | Val chrF++ | Val Geo Mean |
|---|---|---|---|---|---|
| NLLB-200 600M (baseline) | 20.84 | 20.24 | — | — | — |
| NLLB-200 600M + pipeline fixes | 24.95 | 24.54 | 20.00 | 43.66 | 29.55 |
| **NLLB-200 1.3B (best submission)** | **24.87** | **24.41** | ~21 | ~44 | ~28 |

> **Evaluation Metric:**  
> `sqrt(BLEU × chrF++)` — geometric mean of corpus-level BLEU and chrF++.

---

## 🧠 Core Technical Insights

Some of the most important findings from the project:

- Sentence-level alignment dramatically improved translation quality over document-level training.
- The `<gap>` token bug silently removed predictions during decoding.
- chrF++ remained strong while BLEU lagged, suggesting character-level understanding but weak lexical precision.
- Data quality and structural alignment mattered more than model scale.
- Byte-level architectures such as ByT5 may provide advantages for Akkadian morphology and rare-token handling.

---

## 📂 Repository Structure

```text
deep-past-akkadian-mt/
├── notebooks/
│   ├── training/              # LoRA fine-tuning notebooks
│   ├── inference/             # Submission + inference notebooks
│   └── data/                  # Data pipeline and preprocessing
├── reports/
│   └── TECHNICAL_REPORT.md    # Full technical report
└── README.md
```

---

## 📄 Full Technical Report

For a detailed breakdown of:
- The `<gap>` token silent-strip bug and fix
- Why `transformers==4.38.2` was pinned (Group Beam Search offline compatibility)
- chrF++ vs. BLEU diagnostic analysis
- The case for ByT5 as the next architecture
- Custom TPU training loop (replacing HuggingFace Trainer)
- decoding strategies
- experimental results

👉 **[Read the Full Technical Report](reports/TECHNICAL_REPORT.md)**

---

## 🛠️ Tech Stack

```text
Python
PyTorch
HuggingFace Transformers
PEFT / LoRA
SacreBLEU
Kaggle TPU v5e-8
transformers==4.38.2
```

---

## 📚 Acknowledgements

- Kaggle Deep Past Initiative Challenge
- Meta AI — NLLB-200
- HuggingFace Transformers
- PEFT / LoRA
- ORACC Akkadian-English resources
- Old Assyrian Lexicon (eBL)

---

<p align="center">
  Built with curiosity for ancient languages, low-resource NLP, and machine translation research.
</p>

<p align="center">
  <i>Kaggle Deep Past Initiative Challenge — March 2026</i>
</p>
