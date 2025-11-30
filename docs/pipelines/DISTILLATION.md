# NexaCompute Distillation Guide

Complete guide for running knowledge distillation workflows in NexaCompute.

## Overview

**Nexa Distill** transforms raw scientific text into high-quality, falsifiable, and reproducible hypothesis–method pairs for supervised fine-tuning. The engine modularizes teacher generation, filtering, inspection, regeneration, and packaging into final training datasets.

## Architecture

```
nexa_distill/
├── collect_teacher.py        # Generate teacher completions
├── filter_pairs.py            # Clean + filter teacher outputs
├── ui_inspect.py              # Streamlit review interface
├── regenerate_bad.py          # Re-run rejected samples
├── to_sft.py                  # Package final dataset
├── prompts/
│   ├── hypothesis.txt
│   ├── methodology.txt
│   └── rubric.json
├── utils/
│   ├── io.py
│   ├── openai_api.py
│   ├── filters.py
│   ├── texttools.py
│   └── logger.py
└── configs/
    ├── distill_config.yaml
    ├── teacher_models.yaml
    └── filters.yaml
```

## Complete Pipeline

### Stage 1: Prepare Teacher Inputs

Generate teacher input dataset from enhanced prompts:

```bash
# Run analysis notebook to curate teacher inputs
jupyter notebook nexa_data/data_analysis/distill_data_overview.ipynb
```

**Output:** `data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet`

### Stage 2: Collect Teacher Completions

Generate teacher outputs using a strong model (GPT-4, Claude, or Sonnet).

```bash
python -m nexa_distill.collect_teacher \
  --src data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet \
  --dst data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet \
  --teacher openrouter:gpt-4o \
  --max-samples 6000
```

**Input:** Teacher input parquet
**Output:** `data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet`

### Stage 3: Filter Teacher Outputs

Drop weak, incomplete, or low-quality completions.

```bash
python -m nexa_distill.filter_pairs \
  --src data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet \
  --dst data/processed/distillation/filtered/teacher_filtered_v1.parquet
```

**Filtering Rules:**
- Length > 120 chars
- Contains action verbs ("prepare", "simulate", "evaluate", "compare")
- Reject hallucinated citations or broken formatting
- No bracketed references or citation markers

**Output:** `data/processed/distillation/filtered/teacher_filtered_v1.parquet`

### Stage 4: Human Review (Optional)

Streamlit UI for visual inspection and labeling.

```bash
streamlit run nexa_distill/ui_inspect.py \
  -- --src data/processed/distillation/filtered/teacher_filtered_v1.parquet
```

Produces:
- `accepted.jsonl` - Human-approved samples
- `rejected.jsonl` - Samples to regenerate

**Labels stored:** `data/processed/distillation/labels/<date>.jsonl`

### Stage 5: Regenerate Rejected Samples

Re-generate rejected rows via stricter teacher prompts emphasizing falsifiability and reproducibility.

```bash
python -m nexa_distill.regenerate_bad \
  --rejected data/processed/distillation/labels/rejected.jsonl \
  --dst data/processed/distillation/filtered/teacher_regenerated_v1.parquet
```

**Output:** `data/processed/distillation/filtered/teacher_regenerated_v1.parquet`

### Stage 6: Package for Training

Convert all accepted data into SFT-ready JSONL format.

```bash
python -m nexa_distill.to_sft \
  --src data/processed/distillation/filtered/teacher_filtered_v1.parquet \
  --dst data/processed/distillation/sft_datasets/sft_scientific_v1.jsonl
```

**Output:** `data/processed/distillation/sft_datasets/sft_scientific_v1.jsonl`

### Stage 7: Train Student Model

Train student model on distilled dataset.

```bash
python -m nexa_train.distill \
  --dataset data/processed/distillation/sft_datasets/sft_scientific_v1.jsonl \
  --config nexa_train/configs/baseline.yaml \
  --tags distill-v1 scientific-assistant
```

## Implementation Details (Run Scripts)

For production execution, we use specialized scripts managed via tmux sessions.

### Data Generation
- **Script**: `scripts/python/data_processing/run_full_data_gen.py`
- **Purpose**: Generate teacher outputs for full dataset (no max_samples limit)
- **Session**: `data_gen`
- **Config**: `nexa_distill/configs/distill_config.yaml`

### SampleGate Filtering
- **Module**: `nexa_distill/sample_gate.py`
- **Purpose**: Quality gate filtering with judge scores, JSON validation, and safety flags
- **Features**:
  - Minimum judge score threshold (default: 0.80)
  - JSON validity checking
  - Safety flag detection
  - Rejection reason tracking

### Filtering Pipeline
- **Script**: `scripts/python/data_processing/run_filtering.py`
- **Purpose**: Run complete filtering pipeline (basic filters + SampleGate)
- **Session**: `filtering`
- **Outputs**:
  - Filtered dataset: `data/processed/distillation/filtered/filtered_v1.parquet`
  - Rejections: `data/processed/distillation/filtered/rejections.parquet`

### SFT Packaging
- **Script**: `scripts/python/deployment/run_packaging.py`
- **Purpose**: Convert filtered dataset to SFT format (JSONL + Parquet)
- **Session**: `packaging`
- **Outputs**:
  - SFT JSONL: `data/processed/training/sft_dataset.jsonl`
  - SFT Parquet: `data/processed/training/sft_dataset.parquet`

### Training
- **Script**: `scripts/shell/training/run_training.sh`
- **Purpose**: Launch training job (single or distributed)
- **Session**: `training`
- **Config**: `nexa_train/configs/baseline_distill.yaml`

### Orchestration
- **Script**: `scripts/shell/orchestration/launch_pipeline.sh`
- **Purpose**: Launch all pipeline stages in separate tmux sessions.

## Performance Notes

Based on production runs (Nov 2025):
*   **Async Speedup**: Implementing async processing with `ThreadPoolExecutor` (256 workers) yielded a **33x speedup** compared to sequential generation.
    *   100k samples processed in ~3 hours.
*   **Filtering Retention**: Strict post-processing can result in low retention (<1%). Prompt engineering and SampleGate tuning are critical to improve yield.
