# Nexa Infrastructure & Operations

> **Scope**: Provisioning, Scheduling, and Monitoring.
> **Modules**: `nexa_infra`

This module handles the low-level infrastructure required to run the platform, bridging the gap between the software stack and the physical/cloud hardware.

## Core Components

### 1. Provisioning (`nexa_infra/provisioning`)
*   **Cluster Management**: Bootstraps ephemeral GPU clusters (e.g., via Prime Intellect or Slurm).
*   **Tailscale Integration**: Secures node-to-node communication.

### 2. Scheduling (`nexa_infra/scheduling`)
*   **Slurm Adapter**: Submits and monitors jobs on HPC clusters.
*   **Docker**: Manages container lifecycles for consistent execution environments.

### 3. Monitoring (`nexa_infra/monitoring`)
*   **Cost Tracking**: Estimates GPU hours and API usage costs.
*   **Resource Usage**: Tracks CPU/RAM/GPU saturation.

## Containers

The platform uses specialized Docker images for different workloads:
*   `train-heavy.Dockerfile`: For heavy training jobs (Axolotl, CUDA 12.1).
*   `train-light.Dockerfile`: For CPU-bound tasks or lightweight training.
*   `infer.Dockerfile`: For vLLM serving.

## Docker Deployment

Curated container images for Nexa Compute training and inference workloads live here.

### Image Matrix

| Service | Dockerfile | Default Tag | Purpose |
| --- | --- | --- | --- |
| `nexa_light` | `docker/train-light.Dockerfile` | `ghcr.io/nexa/nexa_light:latest` | HF/TRL stack for ≤20B parameter jobs (LoRA, QLoRA, TRL finetunes). |
| `nexa_heavy` | `docker/train-heavy.Dockerfile` | `ghcr.io/nexa/nexa_heavy:latest` | Axolotl, DeepSpeed, FSDP for >20B or multi-node jobs. |
| `nexa_infer` | `docker/infer.Dockerfile` | `ghcr.io/nexa/nexa_infer:latest` | Lean vLLM/TensorRT-LLM inference stack. |

### Common Baseline

All images:
- Pin CUDA 12.1.1 runtime with cuDNN8 on Ubuntu 22.04.
- Use Python 3.11 (`uv` pre-installed for dependency management).
- Install PyTorch from the official cu121 wheel index (no source builds).
- Assume a Hugging Face cache mounted at `/home/runner/.cache/huggingface`.

The legacy monolithic image remains in `docker/Dockerfile` for backward compatibility, but new workflows should stick to the trio above.

### Quick Start

#### 1. Bootstrap from Any Fresh Node

```
bash bin/nexa-bootstrap.sh
```

- Pulls `training-light` by default.
- Mounts the current repo into `/workspace/nexa_compute`.
- Mounts `${HOME}/.cache/huggingface` for reuse across runs.
- Supports overrides via environment variables:
  - `WORK=/custom/workspace bash bin/nexa-bootstrap.sh …`
  - `NVME=/mnt/nvme2 bash bin/nexa-bootstrap.sh …`
  - `SHM_SIZE=32g bash bin/nexa-bootstrap.sh ghcr.io/nexa/training-heavy:latest`

#### 2. One-Liner Helpers

```
# train ≤20B parameter models
python orchestrate.py run train-light

# train >20B / multi-node jobs
python orchestrate.py run train-heavy

# serve inference with vLLM
python orchestrate.py run infer
```

Each helper shells out to `bin/nexa-bootstrap.sh` with the appropriate image and shared-memory size.

#### 3. Direct Docker Run

```
docker run -it --gpus all --rm \
  --shm-size=16g \
  -v $HOME/.cache/huggingface:/home/runner/.cache/huggingface \
  -v /mnt/nvme:/mnt/nvme \
  -v $PWD:/workspace/nexa_compute \
  --env-file .env \
  ghcr.io/nexa/nexa_light:latest
```

For heavy jobs bump `--shm-size` to at least `32g`. `.env` should house WANDB, Hugging Face, AWS, and NCCL configuration.

#### 4. Compose-Based Publish

To build all three curated images from the repo root, use the publish compose file:

```
docker compose -f docker/compose.publish.yaml build --pull
docker compose -f docker/compose.publish.yaml push

DATE=$(date -u +%Y%m%d)
for svc in nexa_light nexa_heavy nexa_infer; do
  docker tag ghcr.io/nexa/${svc}:cu121-py311-pt22 ghcr.io/nexa/${svc}:latest
  docker push ghcr.io/nexa/${svc}:latest
  docker tag ghcr.io/nexa/${svc}:cu121-py311-pt22 ghcr.io/nexa/${svc}:${DATE}
  docker push ghcr.io/nexa/${svc}:${DATE}
done
```

GitHub Actions mirrors this exact flow on every push to `main`, so merges automatically refresh `ghcr.io/nexa/nexa_*` with the variant, latest, and date tags.

### Build & Publish

```
# Train light
docker build -f docker/train-light.Dockerfile -t ghcr.io/nexa/nexa_light:cu121-py311-pt22 .

# Train heavy
docker build -f docker/train-heavy.Dockerfile -t ghcr.io/nexa/nexa_heavy:cu121-py311-pt22 .

# Inference
docker build -f docker/infer.Dockerfile -t ghcr.io/nexa/nexa_infer:cu121-py311-pt22 .
```

Push canonical tags:

```
docker push ghcr.io/nexa/nexa_light:cu121-py311-pt22
docker tag ghcr.io/nexa/nexa_light:cu121-py311-pt22 ghcr.io/nexa/nexa_light:latest
docker push ghcr.io/nexa/nexa_light:latest
```

Repeat for `nexa_heavy` and `nexa_infer`. Add a date tag (`:YYYYMMDD`) for immutable releases.

### Entrypoints

- `docker/entrypoints/vllm_entry.sh` swaps between an interactive shell and the OpenAI-compatible vLLM server depending on the `MODEL` env var.
- Training images drop users into a prepared shell with a non-root `runner` account.

### Environment Expectations

Key environment variables are passed via `.env` or `--env` flags:

- `WANDB_API_KEY`, `HF_TOKEN`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- NCCL tuning: `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`, `NCCL_DEBUG`
- Inference: `MODEL`, `PORT`, `TP`, and any `VLLM_EXTRA_ARGS`

### Caching & Performance Notes

- Persist `/home/runner/.cache/huggingface` across runs to avoid redownloads.
- Attach high-throughput NVMe (`/mnt/nvme`) for checkpoints and datasets.
- Allocate sufficient shared memory (`--shm-size`) for bigger batch sizes.
- Keep CUDA/PyTorch layers near the top of each Dockerfile for cache reuse.
