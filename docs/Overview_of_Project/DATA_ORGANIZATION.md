---
title: Data Organization
slug: overview/data-organization
description: Directory layouts and artifact conventions for NexaCompute datasets.
---

# Data Organization Guide

This document describes the complete data organization structure for reliable querying and running.

## Directory Structure

```
data/
├── raw/
│   └── {project_slug}/                      # Raw input data (gitignored)
│       ├── README.md                       # Raw data overview
│       ├── DATA_INDEX.md                   # Quick reference index
│       ├── *.json / *.jsonl / *.parquet    # Source corpora
│
└── processed/
    └── {project_slug}/                     # Organized processed outputs
        ├── distillation/                   # Teacher, filtered, and SFT datasets
        ├── tool_protocol/                  # Tool-protocol episodes
        ├── training/                       # Training-ready splits
        ├── evaluation/                     # Evaluation predictions/reports
        └── raw_summary/                    # Dataset statistics
```

## Querying Data

### Using the Query Utility

```python
from nexa_data.data_analysis.query_data import DataQuery

query = DataQuery()

# Load teacher inputs
teacher_df = query.get_teacher_inputs(version="v1")

# List available datasets
datasets = query.list_available_datasets()
```

### Direct Path Access

```python
from pathlib import Path

root = Path("/Users/allanmurimiwandia/Nexa_compute")
teacher_path = root / "data" / "processed" / "scientific_assistant" / "distillation" / "teacher_inputs" / "teacher_inputs_v1.parquet"
df = pd.read_parquet(teacher_path)
```

## Versioning

All processed datasets use versioned naming:
- `teacher_inputs_v1.parquet`
- `train_v1.parquet`
- `distillation_manifest_v1.json`

## File Naming Conventions

- **Format**: `{purpose}_{version}.{ext}`
- **Examples**:
  - `teacher_inputs_v1.parquet`
  - `train_v1.parquet`
  - `predictions_run_20251103_143022.parquet`

## Metadata

Each dataset has an accompanying manifest:
- Location: `data/processed/{category}/manifests/`
- Format: JSON with schema, version, row counts, timestamps

## Notebooks

All analysis notebooks are in `nexa_data/data_analysis/`:
- `distill_data_overview.ipynb` - Main distillation analysis
- `query_data.py` - Query utility

## Data Flow

1. **Raw Data** → `data/raw/`
2. **Analysis** → `nexa_data/data_analysis/distill_data_overview.ipynb`
3. **Processed Outputs** → `data/processed/`
4. **Query** → `nexa_data/data_analysis/query_data.py`

## Storage Policy

- **Local Development**: `data/` (gitignored for large files)
- **Production**: `/mnt/nexa_durable/datasets/` on compute nodes
- **Version Control**: Only manifests and metadata in git

