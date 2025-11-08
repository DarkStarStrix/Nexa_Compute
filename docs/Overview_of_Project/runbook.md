---
title: Runbook
slug: overview/runbook
description: Step-by-step operational guide for managing NexaCompute environments and runs.
---

# Operations Runbook

Complete guide for operating the NexaCompute platform.

## Environment Provisioning

1. Install Python 3.11+.
2. Copy `.env` to `.env.local` and adjust endpoints or credentials as needed.
3. Create virtualenv (`python -m venv .venv && source .venv/bin/activate`).
4. Install deps (`pip install -r requirements.txt` or `uv pip install -r requirements.txt`).
5. (Optional) Build Docker image for reproducible runs.

## Data Lifecycle

- **Raw Data:** Place raw datasets under `data/raw/` or sync from remote storage (`scripts/sync_s3.py`).
- **Processed Data:** Preprocessing outputs live in `data/processed/` organized by purpose (distillation, training, evaluation).
- **Metadata:** Dataset metadata dumps to manifests in `data/processed/{category}/manifests/`.

### Querying Data

Use the query utility for reliable data access:

```python
from nexa_data.data_analysis.query_data import DataQuery

query = DataQuery()
teacher_df = query.get_teacher_inputs(version="v1")
pretrain_df = query.get_pretrain_dataset(shard="001")
```

## Running Training

### Basic Training
```bash
python scripts/cli.py train --config configs/default.yaml
```

- Override config values inline (e.g., `--override training.optimizer.lr=0.0001`).
- Distributed run via `scripts/launch_ddp.sh` or by setting `training.distributed.world_size > 1` (the pipeline auto-launches DDP workers).

### Distillation Training
```bash
python -m nexa_train.distill --config nexa_train/configs/baseline.yaml
```

## Evaluating Models

```bash
python scripts/cli.py evaluate --config configs/default.yaml --checkpoint artifacts/checkpoints/checkpoint_epoch0.pt
```

- Metrics saved to `artifacts/runs/<experiment>/metrics.json`.
- Predictions saved when `evaluation.save_predictions` is true.
- Evaluation outputs organized in `data/processed/evaluation/`.

## Hyperparameter Search

```bash
python scripts/cli.py tune --config configs/default.yaml --max-trials 5
```

- Uses random search around optimizer LR and dropout.
- Extend strategy in `scripts/hyperparameter_search.py`.

## Packaging

```bash
python scripts/cli.py package --config configs/default.yaml --output artifacts/package
```

- Produces tarball with model weights, config snapshot, metrics.

## Monitoring

- **TensorBoard:** `tensorboard --logdir logs/`
- **MLflow:** Set `MLFLOW_TRACKING_URI` (or configure via `.env`), pipeline logs params/metrics automatically when enabled.
- **W&B:** Configure `WANDB_API_KEY` in `.env` for automatic logging.

## Run Manifests

Each training run records a manifest with complete metadata for reproducibility and lineage tracking.

### Manifest Schema

Manifests are stored at `runs/manifests/run_manifest.json` or `data/processed/{category}/manifests/` with the following structure:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NexaRunManifest",
  "type": "object",
  "properties": {
    "run_id": {"type": "string"},
    "config": {"type": "string"},
    "run_dir": {"type": "string"},
    "metrics": {
      "type": "object",
      "additionalProperties": {"type": "number"}
    },
    "checkpoint": {"type": ["string", "null"]},
    "created_at": {"type": "string", "format": "date-time"},
    "version": {"type": "string"},
    "dataset": {"type": "string"},
    "model": {"type": "string"}
  },
  "required": ["run_id", "config", "run_dir", "metrics"],
  "additionalProperties": false
}
```

### Manifest Contents

- **run_id:** Unique identifier for the run (format: `{name}_{timestamp}`).
- **config:** Path to the configuration file used.
- **run_dir:** Directory containing run artifacts.
- **metrics:** Final evaluation metrics (accuracy, loss, etc.).
- **checkpoint:** Path to the final model checkpoint.
- **created_at:** ISO 8601 timestamp of run creation.
- **version:** Dataset or model version used.
- **dataset:** Dataset identifier and version.
- **model:** Model architecture identifier.

### Using Manifests

Manifests enable:
- **Reproducibility:** Recreate exact training conditions.
- **Lineage Tracking:** Trace model outputs back to source data.
- **Cost Tracking:** Link runs to compute costs.
- **Leaderboard:** Aggregate metrics across runs.

Parse manifests programmatically:

```python
import json
from pathlib import Path

manifest_path = Path("runs/manifests/run_20251030_213000.json")
manifest = json.loads(manifest_path.read_text())
print(f"Run {manifest['run_id']}: {manifest['metrics']}")
```

## Troubleshooting

- **Logs:** Check logs under `logs/` and `artifacts/runs/<experiment>/`.
- **GPU Visibility:** Verify GPU visibility (`nvidia-smi`).
- **Determinism:** For deterministic issues, set `training.distributed.seed`.
- **Storage:** Verify storage paths are correctly mounted (see `POLICY.md`).
- **Distributed Issues:** Check node health before joining distributed runs.

## Common Workflows

### Full Training Pipeline
1. Prepare data: `python -m nexa_data.prepare --config configs/data.yaml`
2. Train model: `python -m nexa_train.train --config configs/baseline.yaml`
3. Evaluate: `python -m nexa_eval.judge --checkpoint <path>`
4. Package: `python scripts/cli.py package --checkpoint <path>`

### Distillation Pipeline
1. Generate teacher inputs: Run `nexa_data/data_analysis/distill_data_overview.ipynb`
2. Collect teacher outputs: `python -m nexa_distill.collect_teacher`
3. Filter and package: `python -m nexa_distill.filter_pairs`
4. Train student: `python -m nexa_train.distill`

