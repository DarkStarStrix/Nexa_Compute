---
title: Module Structure
slug: /overview/module-structure
description: Directory-by-directory look at NexaCompute modules and their outputs.
---

# NexaCompute Module Structure

This document describes the streamlined module structure where each directory is a distinct submodule for the ML pipeline with no overlap.

## Core Modules

### `nexa_data/` - Data Pipeline
**Purpose:** Data preparation, analysis, augmentation, and feedback loop.

**Submodules:**
- `data_analysis/` - Jupyter notebooks and query utilities for data exploration
- `feedback/` - Feedback loop to improve data based on evaluation weaknesses
- `filters/` - Data filtering utilities
- `loaders/` - PyTorch DataLoader integrations
- `manifest/` - Dataset registry and versioning
- `schemas/` - Data schemas and validation

**Outputs:** All data artifacts in `data/processed/` organized by purpose.

### `nexa_distill/` - Knowledge Distillation
**Purpose:** Transform raw data into high-quality training datasets via teacher-student distillation.

**Components:**
- `collect_teacher.py` - Generate teacher completions
- `filter_pairs.py` - Quality filtering
- `to_sft.py` - Package for supervised fine-tuning
- `ui_inspect.py` - Streamlit inspection UI
- `prompts/` - Prompt templates
- `utils/` - Distillation utilities

**Outputs:** `data/processed/distillation/`

### `nexa_train/` - Model Training
**Purpose:** Training and fine-tuning workflows.

**Components:**
- `train.py` - Main training loop
- `distill.py` - Distillation training
- `backends/` - Training backends (HuggingFace, etc.)
- `models/` - Model registry
- `optim/` - Optimizers and schedulers
- `sweeps/` - Hyperparameter sweeps
- `configs/` - Training configurations

**Outputs:** Model checkpoints, training logs, manifests.

### `nexa_eval/` - Evaluation
**Purpose:** Model evaluation and benchmarking.

**Components:**
- `generate.py` - Generate predictions
- `judge.py` - Score predictions with rubrics
- `analyze.py` - Aggregate metrics and reports
- `reports/` - Evaluation report templates
- `rubrics/` - Evaluation rubrics
- `tasks/` - Evaluation task definitions

**Outputs:** `data/processed/evaluation/`

### `nexa_ui/` - Visualization
**Purpose:** Streamlit dashboards for viewing data and metrics.

**Components:**
- `leaderboard.py` - Main Streamlit app
- `dashboards/` - Dashboard components
- `static/` - Static assets (CSS, etc.)

**Data Sources:**
- Reads from `data/processed/` parquets
- Visualization of evaluation metrics
- Distillation data inspection
- Training statistics

**Usage:**
```bash
orchestrate.py leaderboard  # Launch Streamlit UI
```

### `nexa_infra/` - Infrastructure
**Purpose:** Cluster provisioning, job launching, and orchestration.

**Components:**
- `provision.py` - Cluster provisioning
- `launch_job.py` - Job launching
- `slurm.py` - Slurm integration
- `cost_tracker.py` - Cost tracking
- `sync_code.py` - Code synchronization
- `cluster.yaml` - Cluster configuration

## Data Organization

All processed data follows organized structure:

```
data/
├── raw/              # Raw input data (gitignored)
└── processed/        # Organized outputs by purpose
    ├── distillation/ # Distillation pipeline outputs
    ├── training/     # Training pipeline outputs
    ├── evaluation/   # Evaluation outputs
    └── raw_summary/  # Analysis summaries
```

## Module Boundaries

Each module has clear boundaries:
- **No overlap** - Each directory serves a distinct purpose
- **Artifact-based communication** - Modules communicate via data artifacts, not direct imports
- **Versioned outputs** - All outputs use versioning (v1, v2, etc.)
- **Queryable data** - Use `nexa_data/data_analysis/query_data.py` for reliable access

## Integration Points

- **Orchestration:** `orchestrate.py` provides unified CLI for all modules
- **Data Access:** `nexa_data/data_analysis/query_data.py` for querying processed data
- **UI:** `nexa_ui/leaderboard.py` visualizes data from all modules
- **Manifests:** All modules output manifests for lineage tracking

## Workflow Example

1. **Data Preparation** (`nexa_data/`):
   ```bash
   orchestrate.py prepare_data
   ```

2. **Distillation** (`nexa_distill/`):
   ```bash
   python -m nexa_distill.collect_teacher
   python -m nexa_distill.filter_pairs
   python -m nexa_distill.to_sft
   ```

3. **Training** (`nexa_train/`):
   ```bash
   orchestrate.py launch --config configs/baseline.yaml
   ```

4. **Evaluation** (`nexa_eval/`):
   ```bash
   orchestrate.py evaluate --checkpoint <path>
   ```

5. **Feedback** (`nexa_data/feedback/`):
   ```bash
   orchestrate.py feedback
   ```

6. **Visualization** (`nexa_ui/`):
   ```bash
   orchestrate.py leaderboard
   ```

## Removed Directories

- **`runs/`** - Removed (not needed, all outputs in `data/processed/`)
- **`nexa_feedback/`** - Merged into `nexa_data/feedback/`

## Benefits

1. **Clear separation** - Each module has distinct responsibility
2. **No sprawl** - Organized data structure eliminates confusion
3. **Queryable** - All data accessible via query interface
4. **Visualizable** - UI reads from organized structure
5. **Maintainable** - Easy to understand and extend

