# Architecture Overview

The Nexa Compute platform is organized into modular layers so each concern can evolve independently.

## Layers
- **Configuration**: YAML files parsed into Pydantic schemas, allowing strong validation and programmatic overrides.
- **Data**: Dataset registry and pipeline support raw/processed storage, metadata logging, and batched PyTorch loaders.
- **Models**: Registry exposes pluggable builders. Reference models (MLP, ResNet, Transformer) demonstrate extensibility.
- **Training**: `Trainer` drives the optimization loop with callbacks for logging, checkpoints, and early stopping. Supports AMP, gradient accumulation, and multi-process (DDP) execution.
- **Evaluation**: Metric registry delegates to scikit-learn for rich analytics, writes prediction artifacts, and generates confusion matrix / calibration visualizations.
- **Orchestration**: Pipeline wires together components, persists resolved configs, auto-launches distributed runs, and streams metrics to MLflow when configured.
- **Tooling**: CLI, shell scripts, Docker, and infra templates facilitate repeatable runs locally and at scale.

## Data Flow
1. Load config and seed environment.
2. Build dataloaders through `DataPipeline`.
3. Instantiate model via registry.
4. Configure callbacks and train, emitting checkpoints/logs.
5. Evaluate best model to produce metrics and packaged artifacts.

## Extensibility Points
- Register new datasets via `DatasetRegistry.register`.
- Implement custom models and call `DEFAULT_MODEL_REGISTRY.register`.
- Define new callbacks, metrics, or evaluation exporters without modifying core loop (including adding observability providers).
- Override CLI commands or scripts with project-specific automation.
