# Operations Runbook

## Environment Provisioning
1. Install Python 3.11+.
2. Copy `.env` to `.env.local` and adjust endpoints or credentials as needed.
3. Create virtualenv (`python -m venv .venv && source .venv/bin/activate`).
4. Install deps (`pip install -r requirements.txt` or `uv pip install -r requirements.txt`).
5. (Optional) Build Docker image for reproducible runs.

## Data Lifecycle
- Place raw datasets under `data/raw/` or sync from remote storage (`scripts/sync_s3.py`).
- Preprocessing outputs live in `data/processed/`.
- Dataset metadata dumps to `artifacts/runs/<experiment>/dataset_metadata.json`.

## Running Training
```bash
python scripts/cli.py train --config configs/default.yaml
```
- Override config values inline (e.g., `--override training.optimizer.lr=0.0001`).
- Distributed run via `scripts/launch_ddp.sh` or by setting `training.distributed.world_size > 1` (the pipeline auto-launches DDP workers).

## Evaluating Models
```bash
python scripts/cli.py evaluate --config configs/default.yaml --checkpoint artifacts/checkpoints/checkpoint_epoch0.pt
```
- Metrics saved to `artifacts/runs/<experiment>/metrics.json`.
- Predictions saved when `evaluation.save_predictions` is true.

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
- TensorBoard: `tensorboard --logdir logs/`
- MLflow: set `MLFLOW_TRACKING_URI` (or configure via `.env`), pipeline logs params/metrics automatically when enabled.

## Troubleshooting
- Check logs under `logs/` and `artifacts/runs/<experiment>/`.
- Verify GPU visibility (`nvidia-smi`).
- For deterministic issues, set `training.distributed.seed`.
