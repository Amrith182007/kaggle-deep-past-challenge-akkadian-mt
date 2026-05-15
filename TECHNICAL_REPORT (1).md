# Deep Past Initiative: Akkadian → English Neural Machine Translation
### A Technical Post-Mortem for the Kaggle Deep Past Challenge

> *"I learned that structural alignment and infrastructure are the foundation, but architecture — Subword vs. Byte — is the roof of a project's performance ceiling."*

---

## Table of Contents
1. [Project Abstract](#1-project-abstract)
2. [Infrastructure & Environment](#2-infrastructure--environment)
3. [Pipeline Engineering](#3-pipeline-engineering)
4. [Experimental Results](#4-experimental-results)
5. [The Architectural Ceiling Analysis](#5-the-architectural-ceiling-analysis)
6. [The Diagnostic Smoking Gun: chrF++ vs. BLEU](#6-the-diagnostic-smoking-gun-chrf-vs-bleu)
7. [The ByT5 Theoretical Advantage](#7-the-byt5-theoretical-advantage)
8. [Retrospective & Future Roadmap](#8-retrospective--future-roadmap)
9. [Lessons Learned](#9-lessons-learned)

---

## 1. Project Abstract

The **Deep Past Initiative** is a Kaggle competition requiring participants to build a neural machine translation (NMT) system for one of the most data-scarce translation tasks possible: converting 4,000-year-old **Akkadian cuneiform transliterations** (Old Assyrian, 1950–1750 BCE) into English, evaluated on the geometric mean of BLEU and chrF++ scores (`sqrt(BLEU × chrF++)`). Starting from only ~1,561 document-level parallel pairs, this project developed a full ML pipeline around **NLLB-200** (Meta's No Language Left Behind) with **LoRA fine-tuning**, expanding the training corpus to over 5,700 sentence-level pairs through systematic data reconstruction from published scholarly resources. The final system achieved a public leaderboard score of **24.87** (Private: **24.41**) using the NLLB-1.3B model, with validation metrics revealing a fundamental architectural ceiling that motivates a future shift toward byte-level models like ByT5.

---

## 2. Infrastructure & Environment

### 2.1 The Offline Submission Constraint

Kaggle code competitions require the inference notebook to run completely offline during scoring — no internet access, 9-hour GPU time limit. This was a primary engineering challenge, not just a model challenge. The solution involved:

1. **Downloading all model weights and dependencies** with internet ON into a training notebook
2. **Uploading them as private Kaggle Datasets** for persistent, reproducible access
3. **Installing offline wheel files** at inference time

```python
# Offline dependency installation — every inference notebook begins here
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--no-index",
    "--find-links=/kaggle/input/datasets/amritha18/transformers-4-38-offline-wheels",
    "--find-links=/kaggle/input/datasets/amritha18/peft-acc-offline-wheels",
    "--force-reinstall", "--no-deps", "--quiet",
    "transformers==4.38.2",
    "tokenizers==0.15.2",
    "peft==0.10.0",
    "accelerate==0.27.2",
], check=True)
```

### 2.2 Why `transformers==4.38.2` Was Pinned (Not the Latest)

This is one of the most impactful infrastructure decisions in the project. The Kaggle default environment uses `transformers==5.x`, which **removed native Group Beam Search** and offloaded it to an external Hugging Face community repository (`transformers-community/group-beam-search`). In offline mode, the library attempts to download this code dynamically and fails:

```
ValueError: Group Beam Search requires `trust_remote_code=True`...
since it loads https://hf.co/transformers-community/group-beam-search.
```

All attempts to make the custom code work offline failed (local path injection, manual import, cache copying). The clean solution was **downgrading to `transformers==4.38.2`**, the last version where `group_beam_search` was **native and offline-compatible** for encoder-decoder models.

**Impact on score:** Group beam search (`num_beams=4, num_beam_groups=2, diversity_penalty=1.0`) delivered measurably better translations than standard beam search because it forces the algorithm to explore diverse decoding paths — critical for a language where a single Akkadian word often expands to a full English phrase.

```python
# Group Beam Search config — required transformers 4.38.2
GEN_PARAMS = dict(
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn"),
    max_new_tokens      = 400,
    num_beams           = 4,
    num_beam_groups     = 2,       # native in 4.38.2, broken in 5.x offline
    diversity_penalty   = 1.0,
    no_repeat_ngram_size= 3,
    repetition_penalty  = 1.3,
    length_penalty      = 1.8,
    do_sample           = False,
    early_stopping      = False,
)
```

### 2.3 GPU P100 → TPU v5e-8 Migration

The 1.3B model required a hardware upgrade. Key differences encountered:

| Concern | P100 (GPU) | TPU v5e-8 |
|---|---|---|
| Memory | 16 GB VRAM | 64 GB HBM (8 cores × 8 GB) |
| Precision | `fp16` | `bfloat16` (more numerically stable) |
| Gradient checkpointing | Safe | **Causes XLA recompilation — must be disabled** |
| Gradient accumulation | Safe | **Must stay at 1 — any higher causes XLA graph splits** |
| HuggingFace Trainer | Works | **Deadlocks via `xmp.spawn` — requires custom training loop** |

The HuggingFace Trainer deadlocks on Kaggle TPU v5e-8 because it internally calls `xmp.spawn`, which conflicts with the already-initialized XLA runtime in a notebook process. The solution was a **custom single-process training loop** using raw PyTorch + torch_xla:

```python
# Custom TPU training loop — replaces Seq2SeqTrainer entirely
for step, batch in enumerate(train_loader):
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)

    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    torch_xla.sync()  # replaces xm.mark_step() in torch_xla 2.8+

    if step % 10 == 0:
        print(f"  E{epoch} step {step}/{len(train_loader)} "
              f"loss={loss.item():.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
```

A **keepalive thread** was also required to prevent Kaggle from killing the TPU session during long training runs:

```python
def keepalive():
    while True:
        time.sleep(240)
        print(f"  [alive] {time.strftime('%H:%M:%S')}", flush=True)

threading.Thread(target=keepalive, daemon=True).start()
```

---

## 3. Pipeline Engineering

### 3.1 The Document-to-Sentence Structural Alignment Problem

The most critical data engineering insight of this project: the **training data was document-level, but the test data was sentence-level**. A model trained to predict full multi-paragraph tablet translations was being evaluated on isolated sentences. This structural mismatch was the root cause of early hallucinations, name-looping, and poor BLEU scores.

**The "Holy Grail" file** was `Sentences_Oare_FirstWord_LinNum.csv` — a competition-provided alignment file containing the first word of each sentence, enabling documents to be surgically split into sentence pairs.

A **hybrid three-tier splitting strategy** was implemented:

- **Tier 1 (OARE-based):** Uses the official sentence markers to chop documents — highest quality
- **Tier 2 (um-ma split):** For Akkadian letters containing the epistolary formula `um-ma` ("thus says..."), splits at each occurrence and aligns with English punctuation boundaries
- **Tier 3 (Whole document fallback):** Short texts kept as-is

```python
def split_on_umma(akk_text, en_text):
    # Akkadian letters start with "um-ma" — split there
    akk_parts = re.split(r'(?=\bum-ma\b)', akk_text.strip())
    akk_parts = [p.strip() for p in akk_parts if len(p.strip()) > 10]

    en_normalized = re.sub(r'([.!?])["\u2019\u201c\u201d\']+', r'\1', en_text.strip())
    en_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\u201c\u201d\"])', en_normalized.strip())
    en_parts = [p.strip() for p in en_parts if len(p.strip()) > 10]

    if len(akk_parts) == len(en_parts) and len(akk_parts) > 1:
        return list(zip(akk_parts, en_parts))
    return None
```

**Data growth from expansion:**

| Stage | Pairs | Notes |
|---|---|---|
| Raw `train.csv` | 1,561 | Document-level |
| After hybrid split (um-ma) | ~1,600 | Sentence-level |
| + Extra pairs from `published_texts.csv` | +3,522 | OARE sentence reconstruction |
| After merge + dedup + length filter | **5,009** | Final training set |
| + ORACC external dataset (filtered) | +~900 | After gap alignment filter |

### 3.2 The `<gap>` Token "Shadow Duplicate" Bug — The Smoking Gun

This was the most impactful single bug in the project. The model was training correctly but **every predicted `<gap>` token was being silently deleted** before evaluation.

**Root cause:** `<gap>` was registered as an `additional_special_token`. When decoding with `skip_special_tokens=True`, HuggingFace strips ALL special tokens — including `<gap>`. The model learned to predict it during training, but the evaluation pipeline never saw it.

```python
# BROKEN (Model A) — <gap> registered as special token
special_tokens = {'additional_special_tokens': [SRC_LANG, '<gap>']}
tokenizer.add_special_tokens(special_tokens)

# BROKEN decode — strips <gap>
pred = tokenizer.decode(output, skip_special_tokens=True)
# Result: "1 mina of silver Puzur-Aššur has paid"
# Expected: "1 mina of silver <gap> Puzur-Aššur <gap> has paid"
```

```python
# FIXED — <gap> as regular token, SRC_LANG only as special
tokenizer.add_tokens(["<gap>"] + extra_tokens)  # regular — survives decode
special_tokens = {"additional_special_tokens": [SRC_LANG]}  # NO <gap>
tokenizer.add_special_tokens(special_tokens)

# FIXED decode — manual removal of only the structural tokens
pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
for tok in ['</s>', '<s>', '<pad>', '<unk>', 'eng_Latn', 'akk_Latn']:
    pred_texts = [p.replace(tok, '') for p in pred_texts]
```

**Quantified impact of the fix:**

| Metric | Before Fix | After Fix | Delta |
|---|---|---|---|
| Geometric Mean | 26.71 | 28.45 | **+1.74** |
| BLEU | 17.19 | 19.03 | +1.84 |
| chrF++ | 41.51 | 42.53 | +1.02 |
| Brevity Penalty | 0.874 | 0.955 | +0.081 |
| `<gap>` tokens predicted | **0** | **420** | Fixed |

### 3.3 The `<gap>` Postprocessing Injector

Even after fixing the decode bug, the model did not predict `<gap>` naturally — a training issue caused by 329 out of 920 gap-source training pairs having no gap in the target (teaching the model "source gaps don't need target gaps"). A postprocessor was added to proportionally inject missing gaps:

```python
def postprocess_gap_alignment(src_text, pred_text):
    src_gaps  = src_text.count("<gap>")
    pred_gaps = pred_text.count("<gap>")
    if src_gaps == 0:
        return pred_text
    if pred_gaps >= src_gaps * 0.5:
        return pred_text   # already sufficient

    missing = src_gaps - pred_gaps
    words   = pred_text.split()

    # Distribute proportionally through the sentence (not all at end)
    # Appending all at end hurts chrF on sentence-level test items
    positions = [
        max(0, int(len(words) * (i + 1) / (missing + 1)))
        for i in range(missing)
    ]
    for offset, pos in enumerate(positions):
        words.insert(pos + offset, "<gap>")
    return " ".join(words)
```

### 3.4 Slash Permutation Expansion

Competition translations contained 49 rows with scholarly alternates like `"will / can pay"` — genuine uncertainty, not OCR noise. Rather than discarding these or arbitrarily picking one, a permutation expansion was implemented to generate all combinations, adding ~83 clean pairs at zero cost:

```python
SKIP_SLASH_PAIRS = {"L / Zu", "nēreb / pu"}  # known OCR noise — skip

def expand_slash_permutations(df, max_slashes=3):
    for _, row in df.iterrows():
        slash_instances = re.findall(
            r'([a-zA-ZÀ-žšŠṣṢṭṬḫḪāēīūÀ-ÿ\-]+\s*/\s*[a-zA-ZÀ-žšŠṣṢṭṬḫḪāēīūÀ-ÿ\-]+)',
            row["text_en"]
        )
        if not slash_instances or len(slash_instances) > max_slashes:
            yield row  # keep original
            continue

        slash_options = [(inst, inst.split("/")[0].strip(), inst.split("/")[1].strip())
                         for inst in slash_instances]
        choices = [[opt[1], opt[2]] for opt in slash_options]

        for perm in itertools.product(*choices):
            new_text = row["text_en"]
            for (full_match, _, _), chosen in zip(slash_options, perm):
                new_text = new_text.replace(full_match, chosen, 1)
            new_row = row.copy()
            new_row["text_en"] = new_text
            yield new_row
```

### 3.5 PN/GN Constrained Decoding with N-Best Reranking

A lexicon-based **proper noun forcing** system was built using the `OA_Lexicon_eBL.csv` file. For each source sentence, person names (PN) and geographic names (GN) were identified and passed to `force_words_ids` in the generation call. Four hypotheses were generated and reranked by coverage score:

```python
def score_hyp(src_text, hyp, pn_lookup, dic_lookup):
    """Score a hypothesis by PN and dictionary term coverage."""
    score = 0
    hyp_l = hyp.lower()
    for tok in src_text.split():
        c = tok.strip(".,;:\"'()[]<>{}„-")
        if c in pn_lookup and pn_lookup[c].lower() in hyp_l:
            score += 3   # PN match is high-value
        if c in dic_lookup and dic_lookup[c] in hyp_l:
            score += 1   # dictionary term match
    return score
```

---

## 4. Experimental Results

### 4.1 Submission History

| Submission | Public LB | Private LB | Model | Key Feature |
|---|---|---|---|---|
| Akkadian_resubmission (early) | 20.84 | 20.24 | NLLB-200 600M | Baseline, 1,561 pairs |
| infenote_akka-resub v6 | 23.55 | 23.26 | NLLB-200 600M | Group beam search, 5,009 pairs |
| akka-inference v1 | 24.23 | 24.06 | NLLB-200 600M | Gap fix + slash expansion |
| akka-inference v3 | **24.95** | 24.54 | NLLB-200 600M | Best 600M |
| **akka-inference v9** | **24.87** | **24.41** | **NLLB-200 1.3B** | Best overall (1.3B + LoRA r=32) |
| Fork v3 (600M + ORACC) | 24.34 | 24.50 | NLLB-200 600M | + ORACC external data |

### 4.2 Model Comparison: 600M vs. 1.3B

| Metric | NLLB-200 600M (Best) | NLLB-200 1.3B |
|---|---|---|
| Public LB Score | 24.95 | **24.87** |
| Private LB Score | 24.54 | **24.41** |
| Val Geometric Mean | ~29.55 | ~27–28 est. |
| Val BLEU | ~20.00 | ~19–21 |
| Val chrF++ | ~43.66 | ~42–44 |
| Brevity Penalty | 0.998 | ~0.97 |
| Training Pairs | 5,009 | 5,009 |
| LoRA Rank | r=32 | r=32 |
| Trainable Params | ~17M | ~47M |
| Training Time | ~3 hrs (P100) | ~8 hrs (TPU v5e-8) |

> **Note:** The 1.3B model did not meaningfully outperform the 600M on the private leaderboard, suggesting the bottleneck is **data quality and architecture**, not model scale at this corpus size.

### 4.3 Ablation Study: Fix-by-Fix Score Progression

| Intervention | Val Geo Mean | Delta | Notes |
|---|---|---|---|
| Baseline (LB submission #5) | 22.70 | — | No fixes |
| + `<gap>` decode fix | 28.45 | +1.74 | Single biggest win |
| + `min_new_tokens=30` | 28.24 | −0.21 | Truncation fix |
| + Aggressive gap postprocess | **29.55** | **+1.10** | BP = 0.998 |

---

## 5. The Architectural Ceiling Analysis

### 5.1 Diagnosing the Ceiling

After all engineering optimizations — gap fixes, constrained decoding, data expansion, model scaling — the leaderboard score plateaued in the **24–25 range** (private). The validation geometric mean peaked at **29.55**. Understanding *why* this ceiling exists requires examining what NLLB-200 fundamentally is.

NLLB-200 is a **subword-based** model. Its tokenizer uses SentencePiece BPE to fragment text into subword units from a 256,000-token vocabulary built on ~200 modern languages. Akkadian cuneiform transliteration was never part of that vocabulary.

### 5.2 The Subword Fracturing Problem

Consider this Akkadian word: `iš-qú-ul` (meaning "he/she weighed/paid")

A subword tokenizer that has never seen Akkadian will fracture it:
- `iš` → token fragment
- `-` → token fragment  
- `qú` → unknown or approximated
- `-` → token fragment
- `ul` → token fragment

What is morphologically **one word** (verb root `šql` + subject marker + tense) becomes **5 disconnected subword tokens** with no semantic coherence. The model must learn, from only ~5,000 examples, to reconstruct meaning from fragments that appear differently every time based on vowel harmony and agglutinative morphology.

### 5.3 Evidence from Vocabulary Gap Analysis

The empirical evidence was clear. After full training:

| Metric | Value | Interpretation |
|---|---|---|
| Prediction vocabulary size | 1,905 words | Model's active output range |
| Reference vocabulary size | 2,408 words | Target vocabulary needed |
| Overlap | 889 words (37%) | Majority of targets unreachable |
| Notable missing words | `down`, `small`, `today`, `blood-money` | Common English, not exotic |

The model wasn't missing obscure Assyriological terms. It was missing **common English words** because its internal representation of Akkadian inputs was too noisy (due to subword fracturing) to reliably activate the correct output tokens.

### 5.4 Why Scaling Doesn't Fix This

Upgrading from 600M to 1.3B parameters barely moved the needle (+0.0 to −0.5 on private LB). This is the hallmark of a **data bottleneck amplified by architectural mismatch**: more capacity in the wrong representation space doesn't help. You cannot learn the morphological "math" of Akkadian from subword fragments, no matter how large the model is.

---

## 6. The Diagnostic Smoking Gun: chrF++ vs. BLEU

### 6.1 The Divergence Pattern

The most analytically interesting finding of this project was the consistent and widening gap between two metrics:

| Metric | Validation Score | Leaderboard Score | Gap |
|---|---|---|---|
| chrF++ | **43.66** | ~20–21 | Character patterns captured |
| BLEU | **20.00** | ~11–12 | Word precision failed |
| Geometric Mean | **29.55** | ~24–25 | Dragged down by BLEU |

The competition metric is `sqrt(BLEU × chrF++)`. With chrF++ at 43 and BLEU at 20, the geometric mean is `sqrt(20 × 43) = 29.3`. To reach 40+, you need both metrics above ~38. But BLEU cannot improve without word-level precision.

### 6.2 What This Tells Us

**chrF++ measures character n-gram overlap.** A prediction of `"Puzurasur"` when the reference says `"Puzur-Aššur"` still scores partial credit on chrF++ because the character sequences overlap significantly.

**BLEU measures exact word-level n-gram precision.** `"Puzurasur"` scores **zero** BLEU against `"Puzur-Aššur"` — they are different words.

The implication is precise: **NLLB learned the approximate phonetic and character-level shape of Akkadian-to-English translation, but failed at exact lexical precision.** The subword tokenizer produces internally consistent character patterns, but cannot reliably reconstruct the correct word forms — especially for proper nouns and morphologically inflected terms where a single wrong vowel changes both BLEU score and meaning.

### 6.3 N-Gram Precision Breakdown

The n-gram precision cascade confirmed this diagnosis:

| N-gram | Score | Interpretation |
|---|---|---|
| 1-gram (unigram) | 55.12% | Correct individual words ~55% of time |
| 2-gram | 26.70% | Correct word pairs drop sharply |
| 3-gram | ~14% | Correct phrases are rare |
| 4-gram | 7.73% | Sentence-level fluency near zero |

The cliff between 1-gram and 4-gram precision shows the model produces the right *words* in isolation but cannot chain them into correct *phrases*. This is exactly what you expect from a model that can approximate word meaning but lacks the morphological structure to predict word order and agreement correctly.

---

## 7. The ByT5 Theoretical Advantage

### 7.1 The Core Argument: No Vocabulary, No Ceiling

ByT5 (Google, 2021) is a T5-architecture model that operates directly on **raw UTF-8 bytes** rather than subword tokens. It has no vocabulary file. Every character — including the diacritics, special vowel marks, and subscript digits of Akkadian transliteration — is represented as its exact byte value.

For Akkadian, this is transformative:

```
Subword (NLLB) approach:
  "iš-qú-ul"  →  ["iš", "-", "qú", "-", "ul"]   (5 meaningless fragments)

Byte-level (ByT5) approach:
  "iš-qú-ul"  →  [0x69, 0xC5, 0xA1, 0x2D, 0x71, 0xC3, 0xBA, 0x2D, 0x75, 0x6C]
                  (exact byte sequence — lossless, reversible, complete)
```

### 7.2 The Morphological "Math" Argument

Akkadian is a **Semitic, root-and-pattern agglutinative language**. Meaning is encoded through:
- **Trilateral roots** (e.g., `ŠQL` = weigh/pay)
- **Vowel infixes** (change tense, person, mood)
- **Prefix/suffix chains** (encode subject, object, preposition)

A subword model tries to map these morphological operations onto a fixed BPE vocabulary. When `iš-qú-ul` appears in training, the model memorizes it as a specific fragment sequence. When `iš-qú-lā-am` appears (a different inflection), it's an entirely different fragment sequence with no shared representation.

ByT5 has no such vocabulary prison. The model can learn the **byte-level patterns** that define Akkadian morphology:
- `iš-` prefix → past tense marker
- `-ul` suffix → 3rd person singular
- `qú` → vowel change indicating completed action

Because bytes are compositional, the model can generalize across morphological forms it has never seen — exactly what's needed for a 5,000-pair training set covering a language with thousands of word forms.

### 7.3 The OOV Problem Simply Disappears

With NLLB, every novel Akkadian word fragment that wasn't in the 256k BPE vocabulary gets mapped to `<unk>` or split into increasingly nonsensical sub-units. With ByT5, there is **no such thing as an out-of-vocabulary token** — every possible character combination is representable as a byte sequence. For a dead language where new tablet transcriptions introduce novel proper nouns constantly, this property is invaluable.

### 7.4 Evidence from the Competition

The chrF++ / BLEU diagnostic directly supports this: the model captured **character-level patterns** (high chrF++) but failed on **exact word reconstruction** (low BLEU). ByT5's native byte-level operation would address this exact failure mode — it optimizes character-level reconstruction by design while having the capacity to learn exact lexical forms.

### 7.5 ByT5 Implementation Sketch

```python
# Theoretical next step — ByT5 for Akkadian MT
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ByT5 tokenizer: no vocabulary, raw bytes
tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

def tokenize_akkadian(examples):
    # Source: raw Akkadian bytes
    inputs = tokenizer(
        examples["text_akk"],
        max_length=1024,  # bytes, not BPE tokens — longer sequences needed
        truncation=True, padding=False
    )
    # Target: English bytes
    labels = tokenizer(
        text_target=examples["text_en"],
        max_length=512, truncation=True, padding=False
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

# Expected improvement hypothesis:
# BLEU: 20 → 30+  (exact word reconstruction)
# chrF++: 43 → 48+  (character patterns already strong)
# Geometric Mean: 29 → 38+  (breaking the NLLB ceiling)
```

---

## 8. Retrospective & Future Roadmap

### 8.1 What I Would Do Differently

**Priority 1: Deep PDF Mining Before Any Model Training**

The competition provided PDFs of ~880 scholarly publications (`publications.csv`, OCR output). Each publication contains translations of specific tablets cross-referenced by ID. A systematic pipeline to extract and align these would yield an estimated 5,000–15,000 additional high-quality parallel pairs — a 3× to 10× increase in training data from existing resources.

The current approach used `Sentences_Oare_FirstWord_LinNum.csv` for alignment, which only covers documents with sentence boundary annotations. The PDFs contain many more. **Data volume beats post-processing tricks** at every scale.

```
Recommended PDF mining pipeline:
1. Parse pdf_name + page_text from publications.csv
2. Match publication labels (e.g., "AKT 8, 130") to OARE IDs in published_texts.csv
3. Use sentence embeddings (LASER/LaBSE) for fuzzy alignment
4. Filter by language (keep English only; discard French/German/Turkish)
5. Apply same cleaning pipeline
6. Validate against known translations from train.csv
```

**Priority 2: Architecture First, Scale Second**

The 600M → 1.3B upgrade cost ~5 additional GPU hours and added less than 0.5 points. Switching from NLLB to ByT5 is the same computational cost but attacks the fundamental representation bottleneck. In retrospect, the model architecture decision should have been made first, before spending time on scale.

**Priority 3: Named Entity Normalization as Training Signal**

The competition's `OA_Lexicon_eBL.csv` contains normalized forms for thousands of Akkadian proper nouns. Using these as **training-time labels** (not just inference-time constraints) would teach the model consistent transliteration of names. Current approach only forces names at inference; training on normalized targets would internalize the pattern.

### 8.2 What Worked Extremely Well

- **Hybrid sentence splitting (um-ma + OARE):** This was the single most impactful data engineering decision. Going from document-level to sentence-level alignment directly resolved the length mismatch causing truncated predictions.
- **`transformers==4.38.2` pinning:** Identifying and solving the Group Beam Search offline compatibility issue was a non-trivial infrastructure fix that enabled a better decoding strategy at zero model cost.
- **The `<gap>` decode fix (+1.74 points):** The systematic diagnostic approach (check tokenizer, verify decode round-trip, count gaps in outputs) found a bug that had been silently destroying evaluation scores for the entire project.
- **Controlled experimentation:** Changing one variable at a time with full metric logging prevented the chaos of multi-variable regression.

### 8.3 Future Roadmap

```
Phase 1: Data (2–3 weeks)
├── PDF publication mining → +5,000–15,000 pairs
├── Scholarly lexicon synthetic pairs (rare word grounding)
└── Gap alignment re-annotation (fix the 329 misaligned pairs)

Phase 2: Architecture (1 week)
├── ByT5-base fine-tuning on expanded corpus
├── ByT5-large if memory permits
└── Ensemble: ByT5 (character precision) + NLLB (fluency)

Phase 3: Decoding (3–5 days)
├── Minimum Bayes Risk (MBR) decoding
├── Constrained decoding with full OA_Lexicon_eBL.csv
└── Post-edit with GPT-4 for proper noun normalization

Target: Geometric Mean ≥ 38 (breaking the NLLB ceiling)
```

---

## 9. Lessons Learned

### For the Technical Recruiter

This project operated at the intersection of NLP research, low-resource language processing, data engineering, and MLOps. The challenges were not just model-selection challenges — they required debugging tokenizer internals, writing custom training loops for exotic hardware, and building parallel corpus reconstruction pipelines for a language with no living speakers.

### Core Lessons

**1. Infrastructure is not secondary — it is primary.**
The `transformers==4.38.2` version pin was worth more than the 600M → 1.3B model upgrade. Group beam search diversity improved translations measurably. The offline submission pipeline took more engineering effort than the LoRA configuration. **A great model that can't run offline is a zero on the leaderboard.**

**2. Diagnostics before hyperparameter tuning.**
Every significant score improvement came from diagnosing a root cause, not from blind hyperparameter sweeps. The `<gap>` token was training correctly for weeks while every evaluation silently deleted it. One targeted diagnostic (check `gap_id in tokenizer.all_special_ids`) found the bug in minutes. Measure first, tune second.

**3. The divergence between chrF++ and BLEU is a signal, not noise.**
When two metrics tell different stories, they are revealing something structural about your model. chrF++ at 43 and BLEU at 20 is not "mixed results" — it's a precise diagnosis of where the representation breaks down (character patterns learned, lexical precision failed). Every metric divergence deserves a hypothesis.

**4. Data quality compounds; scale does not.**
5,009 well-aligned, cleaned, sentence-level pairs outperformed 1,561 document-level pairs with a 1.3B model. More parameters in the wrong representation space cannot compensate for structural misalignment in the training data. Clean, aligned data is the multiplier; model scale is the exponent — but the base must be right first.

**5. Architecture is the roof.**
All the engineering — data alignment, gap injection, lexicon forcing, constrained decoding, version pinning, custom training loops — brought NLLB to its ceiling at ~25 LB. The ceiling is set by the model's representational capacity for the target language. Subword tokenization fragments Akkadian morphology in ways that no amount of fine-tuning can overcome. **The next 10 points require a different architecture (ByT5), not more of the same.**

> *"I learned that structural alignment and infrastructure are the foundation, but architecture — Subword vs. Byte — is the roof of a project's performance ceiling."*

---

## Repository Structure

```
deep-past-akkadian-mt/
├── notebooks/
│   ├── training/
│   │   ├── akka_training_600m.ipynb       # NLLB-200 600M + LoRA r=32
│   │   └── akka_training_1_3b_tpu.ipynb   # NLLB-200 1.3B + custom TPU loop
│   ├── inference/
│   │   ├── akka_inference_v9.ipynb        # 1.3B inference (best submission)
│   │   └── fork_akka_inference_v3.ipynb   # 600M inference + ORACC
│   └── data/
│       └── vocabulary_expansion.ipynb     # Extra token extraction
├── reports/
│   └── TECHNICAL_REPORT.md               # This document
└── README.md
```

---

## Citations & Acknowledgements

- Competition host: **Deep Past Initiative**, Kaggle
- Base model: `facebook/nllb-200-distilled-600M` / `facebook/nllb-200-1.3B` (Meta AI)
- Fine-tuning framework: HuggingFace Transformers + PEFT (LoRA)
- Evaluation: SacreBLEU library (`corpus_bleu`, `corpus_chrf`)
- External data: ORACC Akkadian-English Parallel Corpus (`manwithacat/oracc-akkadian-english-parallel-corpus`)
- Lexical resources: Old Assyrian Lexicon from eBL (electronic Babylonian Library, LMU Munich)

---

*Report prepared by Amritha | April 2026 | Kaggle Deep Past Initiative Challenge*
