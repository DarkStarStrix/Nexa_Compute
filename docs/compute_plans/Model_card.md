# Nexa-Spec Foundation Model — Model Card (Planning Draft)

This document serves as the **master reference** for the architecture, data strategy, compute plan, and training schedule for the Nexa-Spec spectral foundation model. It consolidates all decisions into a single technical specification that will guide pretraining and reproducibility.

---

# 1. Model Overview

**Model Name:** Nexa-Spec
**Type:** Encoder-only Transformer for MS/MS spectral representation learning
**Scale Targets:**

* **v1:** Nexa-Spec-3B (prototype training)
* **v2:** Nexa-Spec-7B (full-scale run)

**Training Regime:**
Self-supervised pretraining on MS/MS spectra using:

* Masked Peak Modeling (primary)
* Retention Order Prediction (secondary)
* Optional: Contrastive Spectrum Embedding (retrieval)

The model learns a general-purpose spectral embedding that supports fingerprint prediction, property prediction (RT/IM), and structure generation (two-stage pipeline).

---

# 2. Architecture Specification

## 2.1 Input Representation

Each MS/MS spectrum is converted into a sequence of **peak tokens**:

* Sorted by *m/z*
* Maximum peaks: **256** (preprocessed)
* Special tokens:

  * `[CLS]` (global)
  * `[PREC]` (precursor)
  * `[MASK]` (for masked peak modeling)

### Token Embedding Components

* *m/z* → Fourier positional embedding → linear projection → `d_model`
* Intensity → small MLP → `d_model`
* Token type embedding (peak/precursor/masked)
* Final token embedding: **vector in ℝ^{d_model}**

---

# 3. Transformer Backbone

## 3.1 Nexa-Spec-3B (Pilot)

* Layers: **36**
* Hidden size: **2304**
* FFN size: **9216**
* Heads: **32**
* Attention: Local window (64 peaks) + global tokens
* Params: **~2.7–3B**

## 3.2 Nexa-Spec-7B (Final)

* Layers: **60**
* Hidden size: **3072**
* FFN size: **12,288**
* Heads: **48**
* Attention: Local window (64 peaks) + global tokens
* Params: **~7.1–7.3B**

### Attention Structure

Each layer uses:

* **Local attention**: limited receptive field in m/z
* **Global attention**: `[CLS]` and `[PREC]` attend everywhere

This hybrid design preserves fragment-level context while enabling spectrum-level reasoning.

---

# 4. Pretraining Heads

## 4.1 Masked Peak Modeling (Primary)

* Randomly mask 15–30% of peaks
* Predict:

  * Discretized m/z bin (e.g. 4096 bins)
  * Discretized intensity bin
* Loss: Cross-entropy on m/z + intensity bins

## 4.2 Retention Order Prediction (Secondary)

* Input: Pair of spectra concatenated with `[SEP]`
* Predict whether A elutes before B
* Loss: Binary cross-entropy

## 4.3 Optional: Contrastive Embeddings

* InfoNCE loss over `[CLS]` projections
* Used later for retrieval fine-tuning

---

# 5. Dataset Specification

## 5.1 Data Volume

* Total spectra: **550 million**
* Train/val/test split: **98% / 1% / 1%**
* Train spectra: **~539 million**

## 5.2 Tokenization

* ~256 tokens per spectrum

Token count per epoch:

* `539M × 256 ≈ 1.38 × 10¹¹ tokens`

Token count for 3 epochs:

* `≈ 4.1 × 10¹¹ tokens`

This is sufficient for compute-optimal training of a **7B model**, per Chinchilla scaling (10–20×P tokens).

---

# 6. Training Schedule

## 6.1 Compute

* **16× H100 SXM5** on a single VM
* NVLink interconnect between GPUs
* Local NVMe for data shards
* Optional InfiniBand for multi-node scaling

## 6.2 Parallelism Strategy

* **FSDP (Fully Sharded Data Parallel)** over all 16 GPUs
* 1 process per GPU: `WORLD_SIZE = 16`
* Mixed precision: **bf16**
* Gradient checkpointing enabled

No tensor or pipeline parallelism required at 7B scale.

## 6.3 Batch Size Assumptions

Example:

* Per-GPU batch size: **32 spectra**
* Global batch: `16 × 32 = 512 spectra`

Steps per epoch:

* `539M / 512 ≈ 1.05M steps`

3 epochs:

* **~3.1M steps total**

## 6.4 Optimizer & Schedule

* Optimizer: **AdamW**
* β1 = 0.9, β2 = 0.95
* Weight decay = 0.1
* LR schedule: Cosine
* Peak LR:

  * 3B → 1.5e-4
  * 7B → 1.2e-4
* Warmup: 5k–10k steps

---

# 7. Data Pipeline Overview

## 7.1 Data Flow

```
HF Raw HDF5 → CPU VM (AstroData) → Arrow Shards → Wasabi Bucket → H100 VM → Training
```

## 7.2 Storage

* Wasabi bucket with versioned dataset folders
* Shards (~2GB each) placed in:

```
processed/gems_v1/shards/{train,val,test}
```

* Manifest auto-generated per version

## 7.3 Dataset Versioning

* v1 → sanity build
* v2 → pipeline refinements
* v3 → production-quality dataset
* Full training run uses **gems_v3**

---

# 8. Training Stability & Monitoring

## Must-track metrics

* Total loss
* Masked peak loss
* Retention order loss
* LR
* Grad norms
* Embedding variance (optional)
* Validation masked loss (rare checks)

## Early failure signals

* Loss NaN → lower LR
* Attention divergence → reduce batch size
* Unbalanced losses → reweight RO head

---

# 9. Post-Pretraining Heads

Once SSL pretraining completes:

* Fingerprint classification (2048/4096 bits)
* RT regression (heteroscedastic)
* CCS regression
* Contrastive embedding fine-tuning
* Structure generation (two-stage: fingerprint → SMILES)

The pretrained encoder is frozen/unfrozen as needed.

---

# 10. Summary

Nexa-Spec is a **7B-parameter spectral foundation model** trained on **>4×10¹¹ tokens** from MS/MS spectra. Built as a deep encoder-only Transformer with domain-specific attention and masking objectives, it is designed to learn universal spectral embeddings usable for:

* molecular fingerprinting
* retention time prediction
* ion mobility prediction
* spectral retrieval
* de novo structure generation

This model card captures the decisive architecture, data, and compute plan needed to execute the full pretraining run.
