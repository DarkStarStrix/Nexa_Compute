# Storage Policy

## Directories
- `data/raw/`: ingest-only assets. Treated as ephemeral and excluded from version control.
- `data/processed/`: feature-ready datasets, regenerated on demand.
- `runs/`: canonical location for manifests, logs, and checkpoints.
  - `runs/manifests/`: JSON manifests, cluster state, and cost reports.
  - `runs/logs/`: structured logs streamed during training/eval.
  - `runs/checkpoints/`: persisted model weights.
- `artifacts/`: transient outputs (metrics, packaged models). Ignored by git.

## Retention
- Raw/processed data should be backed by remote object storage with lifecycle rules (30-day retention).
- Run manifests retained indefinitely for provenance.
- Checkpoints older than the top-K best can be pruned automatically.

## Access Control
- Paths respect `.env` overrides (`NEXA_DATA_DIR`, `NEXA_ARTIFACT_DIR`).
- Production deployments mount object storage buckets using IAM roles scoped per environment.
