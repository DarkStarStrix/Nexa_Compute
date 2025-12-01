# Nexa Infrastructure & Operations

> **Scope**: Provisioning, Scheduling, Monitoring, and Containerization.
> **Modules**: `nexa_infra`

The `nexa_infra` module is the foundation of the platform. It abstracts the complexity of underlying hardware—whether local development machines, ephemeral cloud GPU nodes, or HPC Slurm clusters—providing a uniform API for job execution and resource management.

## 1. Job Scheduling & Launching

### The Launcher (`nexa_infra/operations/launch.py`)
This is the high-level switching logic that decides *where* and *how* a job runs.

*   **Local Training**: Launches `nexa_train` directly in the current process or subprocess.
*   **Distributed Training (DDP)**: Uses `torchrun` (via `scripts/launch_ddp.sh`) to spin up multiple processes on a single or multiple nodes. It handles environment variable propagation (`MASTER_ADDR`, `RANK`, `WORLD_SIZE`).
*   **Slurm Sweeps**: Interfaces with the Slurm scheduler to submit job arrays for hyperparameter optimization.

### Slurm Integration (`nexa_infra/scheduling/slurm.py`)
Automates the creation of complex `sbatch` scripts.

*   **Parameter Expansion**: Takes a `SweepDefinition` (grid of parameters) and performs a Cartesian product to generate unique configurations for every job in the array.
*   **Spec Generation**: Writes a `spec.json` file that maps `SLURM_ARRAY_TASK_ID` to specific command-line arguments. The Python worker script reads this spec to configure itself at runtime.
*   **Resource definition**: Maps high-level requests (nodes, gpus) to Slurm directives (`#SBATCH --partition`, `#SBATCH --gpus-per-node`).

## 2. Containerization

NexaCompute enforces reproducible environments via a strict 3-tier container strategy.

### Image Matrix

| Service | Dockerfile | Default Tag | Purpose |
| --- | --- | --- | --- |
| `nexa_light` | `docker/train-light.Dockerfile` | `ghcr.io/nexa/nexa_light:latest` | **Standard Training**: HF/TRL stack for ≤20B models. Optimized for single-node jobs. |
| `nexa_heavy` | `docker/train-heavy.Dockerfile` | `ghcr.io/nexa/nexa_heavy:latest` | **Scale Training**: Axolotl, DeepSpeed, FlashAttention-2. Includes compilation tools for FSDP. |
| `nexa_infer` | `docker/infer.Dockerfile` | `ghcr.io/nexa/nexa_infer:latest` | **Inference**: Lean vLLM/TensorRT-LLM stack. Minimal dependencies for fast startup. |

### Runner (`nexa_infra/containers/runner.py`)
A Python wrapper around the Docker CLI.
*   **Volume Mounting**: Automatically mounts `/workspace`, `/mnt/nvme` (fast storage), and `~/.cache/huggingface` (model cache).
*   **User Mapping**: Maps the host user ID to the container user to prevent file permission issues on shared volumes.
*   **Bootstrap**: Uses `bin/nexa-bootstrap.sh` to initialize the environment inside the container.

## 3. Monitoring & Cost

### Cost Estimation (`nexa_infra/monitoring/costs.py`)
Provides financial visibility into large-scale runs.

*   **Estimation**: Before launching a Slurm sweep, it calculates: `Nodes × GPUs/Node × Est. Duration × Hourly Rate`.
*   **Logging**: Persists cost manifests (`cost_<run_id>.json`) to `runs/manifests/`.
*   **Aggregation**: Can summarize total spend across a campaign.
*   **Defaults**: Includes pricing tables for common GPUs (H100, A100, RTX 4090).

### Resource Monitoring
*   Standard Prometheus/Grafana integrations (via `nexa_infra/monitoring/config/prometheus.yml`) track GPU saturation, VRAM usage, and temperature.

## 4. Operations

### Sync (`nexa_infra/operations/sync.py`)
Critical for remote development.
*   **`sync_repository`**: Uses `rsync` to intelligently push code from a local laptop to a remote cluster head node. It excludes heavy artifacts (`runs/`, `__pycache__`) to ensure sub-second sync times.

### Teardown (`nexa_infra/provisioning/teardown.py`)
*   Automates the cleanup of ephemeral resources (e.g., Terraform state) after a campaign finishes.

## Usage Examples

**Launch a Slurm Sweep:**
```python
from nexa_infra.operations.launch import launch_slurm_sweep
from pathlib import Path

config = Path("nexa_train/sweeps/random_search.yaml")
artifacts = launch_slurm_sweep(config, submit=True)
print(f"Submitted job array {artifacts.job_count} jobs")
```

**Run a Container:**
```python
from nexa_infra.containers.runner import run_container

# Launches an interactive shell in the heavy training container
run_container("train-heavy")
```
