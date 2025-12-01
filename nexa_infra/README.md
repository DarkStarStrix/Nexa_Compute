# Nexa Infra

> 📚 **Full Documentation**: [docs/pipelines/INFRASTRUCTURE.md](../../docs/pipelines/INFRASTRUCTURE.md)

## Overview

The `nexa_infra` module is the backbone of the Nexa Compute platform. It abstracts the complexities of underlying hardware and scheduling systems, providing a unified API for:
*   **Provisioning:** Managing cluster resources (though mostly declarative via Terraform/Ansible in practice).
*   **Scheduling:** dispatching jobs to Slurm clusters or local runners.
*   **Containerization:** Bootstrapping standard environments (`train-light`, `train-heavy`, `infer`).
*   **Monitoring:** tracking costs and resource usage.

## Key Components

### `operations/launch.py`
The high-level interface for starting jobs. It decides whether to run a job locally, submit it to a scheduler, or launch a distributed training session.

#### Functions
*   `launch_training_job(config_path: Path, distributed: bool, ...) -> ArtifactMeta`
    *   Starts a training run. If `distributed=True`, it invokes the `torchrun` / DDP launcher scripts.
*   `launch_slurm_sweep(config: Path, submit: bool) -> SlurmBatchArtifacts`
    *   Converts a sweep configuration into a Slurm job array script and optionally submits it via `sbatch`.

### `scheduling/slurm.py`
Generates complex Slurm batch scripts for hyperparameter sweeps.

#### Classes
*   `prepare_slurm_batch(...)`
    *   Takes a `SweepDefinition` (parameters to search) and a `SlurmJob` (resource constraints) and produces a self-contained job array script. It handles parameter expansion (grid search) and environment injection.

### `containers/runner.py`
Manages the execution of Docker/Singularity containers.

#### Functions
*   `run_container(target: str, ...)`
    *   Pulls and runs a curated container image (e.g., `train-heavy` for large-scale training). It mounts necessary workspace and data volumes automatically.

### `monitoring/costs.py`
Provides estimation and logging for compute costs.

#### Functions
*   `estimate_batch_cost(...)`
    *   Calculates the projected cost of a job based on node count, GPU type, and duration. Useful for budgeting large sweeps.

### `operations/sync.py`
Utilities for synchronizing code and artifacts between local development environments and remote clusters.

#### Functions
*   `sync_repository(destination: str, ...)`
    *   Uses `rsync` to push the current codebase to a remote path, ensuring the cluster runs the latest version.
