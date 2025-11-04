# NexaCompute Quick Start

This guide provides essential information for getting started with NexaCompute. For detailed documentation, see the full documentation files.

## Core Concepts

### Data Flow
1. **Raw Data** → `data/raw/` (JSON, JSONL, Parquet)
2. **Processed Data** → `data/processed/` (organized by purpose)
3. **Training** → Uses processed datasets
4. **Evaluation** → Generates metrics and reports
5. **Feedback Loop** → Improves data based on eval results

### Key Modules
- **`nexa_data/`** - Data preparation and pipelines
- **`nexa_train/`** - Training and distillation
- **`nexa_eval/`** - Evaluation and benchmarking
- **`nexa_distill/`** - Knowledge distillation pipeline
- **`nexa_infra/`** - Infrastructure and orchestration

### Storage Structure
```
data/
├── raw/              # Raw input data (gitignored)
├── processed/        # Organized outputs
│   ├── distillation/
│   ├── training/
│   ├── evaluation/
│   └── raw_summary/
```

## Common Workflows

### Distillation Pipeline
1. Generate teacher inputs: Run `nexa_data/data_analysis/distill_data_overview.ipynb`
2. Collect teacher outputs: `python -m nexa_distill.collect_teacher`
3. Filter and package: `python -m nexa_distill.filter_pairs`
4. Train student: `python -m nexa_train.distill`

### Training Workflow
1. Prepare data: `python -m nexa_data.prepare`
2. Train model: `python -m nexa_train.train`
3. Evaluate: `python -m nexa_eval.judge`

## Documentation Index

- **Architecture**: `architecture.md`
- **Data Format**: `DATA_FORMAT.md`
- **Storage Policy**: `STORAGE_POLICY.md`
- **Distillation**: `Nexa_distill.md`
- **Evaluation**: `EVAL_FRAMEWORK.md`, `eval-and-benchmarking.md`
- **Operations**: `runbook.md`

## Getting Help

- Check `runbook.md` for operational procedures
- Review `STORAGE_POLICY.md` for data organization
- See `DATA_FORMAT.md` for schema details

