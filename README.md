# Nexa Compute Training Platform

A modular framework for running large-scale ML training jobs with reproducible workflows, distributed execution, and experiment tracking. The repository bundles data ingestion, model factories, training orchestration, evaluation, and operational tooling.

## Key Features
- Declarative YAML configuration validated via Pydantic schemas.
- Data pipeline abstraction with dataset registry and metadata logging.
- Model registry with reference MLP, ResNet, and Transformer architectures.
- Trainer with callbacks (logging, checkpointing, early stopping) and AMP support.
- Evaluation suite with rich metrics and artifact export.
- Automation scripts for environment setup, data prep, training, evaluation, tuning, and packaging.
- Docker + distributed launch helpers for scaling to multi-node clusters.

## Repository Layout
```
nexa-compute/
├── orchestrate.py         # High-level CLI entrypoint
├── nexa_infra/            # Provisioning, launch, and cost tracking modules
├── nexa_data/             # Data prep/augmentation/distillation utilities
├── nexa_train/            # Training orchestration, configs, sweeps, optim tooling
├── nexa_eval/             # Evaluation + reporting stack
├── nexa_feedback/         # Weakness analysis & feedback generation
├── nexa_ui/               # Leaderboard + dashboard assets
├── runs/                  # Manifests, logs, checkpoints
├── docs/                  # Architecture, storage, schema, and safety docs
├── src/nexa_compute/      # Core library (config, data, models, training, eval)
├── scripts/               # Supporting automation scripts
├── tests/                 # Unit & integration tests
└── pyproject.toml         # Packaging metadata
```

## Quick Start
1. **Create environment**
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
2. **Provision & prepare data (optional)**
   ```bash
   python orchestrate.py provision --bootstrap
   python orchestrate.py prepare-data --config nexa_train/configs/baseline.yaml
   ```
3. **Run smoke training**
   ```bash
   python orchestrate.py launch --config nexa_train/configs/baseline.yaml
   ```
4. **Evaluate checkpoint**
   ```bash
   python orchestrate.py evaluate --config nexa_train/configs/baseline.yaml
   ```

## Make Targets
```
make install       # install dependencies into current env
make lint          # run static analysis (ruff + mypy)
make test          # run unit tests
make train         # run default training config
make evaluate      # run evaluation for default config
make package       # create tarball with exported model artifacts
```

## Docker
```
docker build -t nexa-compute:latest -f docker/Dockerfile .
docker run --gpus all -v $(pwd):/workspace nexa-compute:latest python scripts/cli.py train --config configs/default.yaml
```

## Distributed Launch (example)
```
bash scripts/launch_ddp.sh --config configs/distributed.yaml --nodes 2 --gpus 4
```

## Experiment Tracking
- TensorBoard logs saved under `logs/`.
- MLflow-ready metadata (set `MLFLOW_TRACKING_URI` env var and the pipeline will auto-log runs).

## Contributing
- Follow style checks via `make lint` before submitting PRs.
- Update `Spec.md` and `docs/` when adding new functionality.
