# Evaluation Specification: Spectrum-to-Structure Foundation Model

## Overview
This document defines the evaluation suite for a 7B-scale spectrum-to-structure model designed to generate molecular structures directly from raw MS/MS spectra. The evaluation framework is split into:

- **ML Metrics**: Standard machine learning measures for structured generation and retrieval.
- **Domain-Specific Metrics**: Chemically grounded evaluations assessing scientific validity and interpretability.
- **Competitive Landscape**: Models in this space and the thresholds they report.
- **Adoption Targets**: Evaluation thresholds aligned with credibility, industry adoption, and viral utility.

---

## Part 1: Machine Learning Metrics

### 1.1 Core Training Metrics
- **Total Loss**: Combination of SMILES token loss, fingerprint BCE, and contrastive loss (if applicable).
- **Per-token Cross Entropy / Perplexity**
- **% Valid SMILES / SELFIES**
- **% Formula-consistent predictions**

### 1.2 Structured Generation Metrics
Evaluated per test spectrum across top-k predicted structures.

- **Top-k Exact Accuracy** (Top-1, Top-10)
- **Top-k Tanimoto Similarity** (using Morgan fingerprints, radius 2)
- **Top-k MCES Distance** (maximum common edge subgraph, lower is better)

### 1.3 Embedding & Retrieval Metrics
Using a candidate pool (|C| ≤ 256), perform retrieval from spectrum embeddings.

- **HitRate@1, @5, @20**
- **MCES@1**, **Tanimoto@1** of retrieved molecule
- **Mean Reciprocal Rank (MRR)**
- **Correlation between embedding similarity and structural similarity**

### 1.4 Calibration & Reliability
- **Expected Calibration Error (ECE)** on SMILES scores, fingerprint scores
- **PICP@90** (Prediction Interval Coverage)
- **Reliability Diagrams**

### 1.5 Generalization Metrics
Compute all metrics sliced by:
- Instrument type (Orbitrap, QTOF, etc.)
- Collision energy buckets (low / mid / high)
- Ionization mode (positive / negative)
- Chemical class (ClassyFire, NPClassifier)

---

## Part 2: Domain-Specific Evaluation Metrics

### 2.1 Precursor Mass & Formula Consistency
- **Mass Error (ppm)**: < 5 ppm (Orbitrap), < 15 ppm (QTOF)
- **Formula Accuracy**: Exact match and ΔElement distributions

### 2.2 Fragmentation Explainability
- **% Peaks Explained** (any matching fragment)
- **% Intensity Explained**
- **# Unexplained High-Intensity Peaks**

### 2.3 Adduct & Ion Mode Validity
- **Adduct-Compatible Predictions (%)**
- **Incorrect charge states / isotope violations (%)**

### 2.4 Chemical Class and Scaffold Correctness
- **ClassyFire Superclass Accuracy (%)**
- **Class Accuracy (%)**
- **Murcko Scaffold Match Rate**
- **Functional Group Recall (%)**

### 2.5 Candidate Reduction / Workflow Gains
- **Candidate Reduction Factor**: From SIRIUS/MSNovelist set size to filtered top-k
- **Avg. Time to Annotation** in simulated workflows (spectra/hr/chemist)

### 2.6 Robustness Across Spectral Conditions
- Evaluate fragmentation metrics, class consistency across:
  - Ionization modes (positive, negative)
  - Collision energies
  - Instrument types

---

## Part 3: Competitive Landscape

### De Novo Structure Models
- **MSNovelist**: Top-1 ~7–39% in constrained settings, strong fragment scoring
- **Spec2Mol**: Focused on similarity; MW/formula errors reported
- **Test-Time Tuned LM (2025)**:
  - NPLIB1: Top-1 ≈ 16.8%, Tanimoto ≈ 0.62, MCES ≈ 6.5
  - MassSpecGym: Top-1 ≈ 2.8%, Tanimoto ≈ 0.45, MCES ≈ 11.9
- **DIFFMS, MADGEN, MS-BART**: Range of 2–4% Top-1, often higher on similarity metrics

### Embedding & Retrieval Models
- **MIST, MS2Query, Spec2Vec**: ~40–60% HitRate@1 on curated tasks, widely used in metabolomics platforms

---

## Part 4: Adoption Thresholds

### Tier 1: "Taken Seriously"
- Top-10 Accuracy ≥ 5–7% (MassSpecGym)
- Tanimoto@1 ≥ 0.50
- HitRate@1 ≥ 35–45%
- ≥60% peaks/intensity explained
- Formula match ≥90%

### Tier 2: "Turning Point"
- Top-10 Accuracy ≥ 10–12%
- Top-1 Tanimoto ≥ 0.55, MCES ≤ 9
- HitRate@1 ≥ 50%
- Fragment explainability ≥70% intensity
- Candidate reduction ≥10–15×
- Class accuracy ≥80%

### Tier 3: "Viral"
- Top-1 Accuracy ≥ 8–10%
- Top-10 Accuracy ≥ 20%+
- Tanimoto@1 ≥ 0.60, MCES ≤ 8
- HitRate@1 ≥ 60%, @20 ≥ 95%
- Fragment explainability ≥80% intensity
- Candidate reduction ≥30×
- 5–10× annotation rate improvement in practice

---

## Integration with Atlas++
Atlas++ should log per-spectrum:
- Raw spectrum metadata
- Inference embeddings
- Predicted SMILES, fingerprints
- Eval metrics:
  - Top-k accuracy, Tanimoto, MCES
  - Mass error, fragmentation explainability
  - Retrieval rank, hit quality
  - Calibration / reliability plots

All metrics should be sliceable across:
- Instrument / ion mode / CE
- Dataset (MassSpecGym, NPLIB1, in-house)
- Adduct type

Versioned evaluation runs should produce nightly:
- Metric trends
- Fail mode diagnostics
- Leaderboard by model checkpoint / config

---

## Reproducibility & Reporting
Each evaluation run must log:
- Model SHA, data split version, tokenizer
- Instrumental metadata used (ion mode, CE, etc.)
- Canonical SMILES normalization config

All benchmarks should be aligned to:
- MassSpecGym (de novo, retrieval)
- NPLIB1 (semi-closed, high-confidence)

---

## References
- MassSpecGym Benchmark Dataset
- Test-Time Tuned LM (Mismetti et al., 2025)
- MSNovelist (Stravs et al., 2022)
- Spec2Mol (Litsa et al., 2023)
- DIFFMS, MS-BART, MADGEN Papers (2023–2025)
- Spec2Vec, MS2Query (Huber et al.)
- Atlas++ Internal Spec【87†Atlas_PlusPlus_Technical_Spec.pdf】

