# Nexa Train

> 📚 **Full Documentation**: [docs/pipelines/TRAINING.md](../../docs/pipelines/TRAINING.md)

## Overview

The `nexa_train` module is the engine for model training and fine-tuning within the Nexa ecosystem. It abstracts over specific training frameworks (backends) to provide a unified interface for:
*   **Supervised Fine-Tuning (SFT)**
*   **Knowledge Distillation**
*   **Distributed Training** (FSDP, etc.)

It integrates deeply with `nexa_compute` for artifact management, ensuring that every training run produces versioned, reproducible checkpoints.

## Key Components

### `train.py`
The primary CLI entry point for executing training pipelines. It handles configuration loading, overrides, pipeline execution, and artifact registration.

#### Functions
*   `run_training_job(config_path: Path, overrides: Optional[List[str]] = None) -> ArtifactMeta`
    *   Initializes a `TrainingPipeline` from a config, runs it, and wraps the resulting checkpoint in a standard artifact.
*   `parse_args()`
    *   Parses command-line arguments, mapping simplified flags (e.g., `--model-size`, `--gpus-per-node`) to complex configuration overrides.

### `launcher.py`
A unified launcher script that dispatches jobs to specific training backends based on YAML configuration.

#### Functions
*   `main(argv: Sequence[str] | None = None) -> int`
    *   Reads a launcher config, resolves backend parameters (allowing for CLI overrides), and invokes the specified backend function.

### `distill.py`
Connects the training module to the distillation data flow.

#### Functions
*   `distill_teacher(config_path: Path, checkpoint: Optional[Path], output_dir: Path) -> Path`
    *   Runs inference using a student model configuration to generate probabilities, then materializes these into a format suitable for distillation analysis.

### `backends/hf.py`
The backend implementation for Hugging Face Trainer.

#### Functions
*   `run(params: Dict[str, Any] | None = None) -> Dict[str, Any]`
    *   Normalizes parameters and executes a training run using `nexa_compute.training.hf_runner`.

### `optim/schedulers.py`
Provides custom learning rate scheduler implementations.

#### Functions
*   `build_scheduler(name: str, optimizer: optim.Optimizer, **kwargs)`
    *   Factory function to create schedulers. Currently supports `cosine_warmup`.

### `utils.py`
Shared utilities for config management.

#### Functions
*   `load_training_config(...)`
    *   Wrapper around the core config loader.
*   `save_run_manifest(run_dir: Path, manifest: dict) -> Path`
    *   Writes a JSON manifest describing the training run's metadata.
