# NexaCompute Infrastructure Overview

**Author:** Allan  
**Date:** November 1, 2025  
**Status:** v1.1 — Infra stable, distillation-ready

---

## Introduction

NexaCompute is designed as a unified, reproducible, and modular ML pipeline for developing distilled scientific assistant models. This document provides a comprehensive overview of the infrastructure, outlining the repository structure, lifecycle, storage, job orchestration, distributed training strategy, and practical operational guidelines.  
The primary goal of this setup is to make every step *boring to run* — predictable, observable, and easily auditable.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Environment Setup](#environment-setup)
3. [Storage Hierarchy](#storage-hierarchy)
4. [Data, Training & Evaluation Scripts](#scripts)
5. [Manifests & Logging](#manifests--logging)
6. [Pipeline Workflow](#pipeline-workflow)
7. [Job Launch System](#job-launch-system)
8. [Distributed Training Specification](#distributed-training-specification)
9. [Operational Practices](#operational-practices)
10. [Scaling Roadmap](#scaling-roadmap)

---

## Repository Structure

```
nexa-compute/
├── nexa_core/
│     └── manifest.py
├── nexa_data/
│     ├── make_distill_dataset.py
│     ├── generate_from_teacher.py
│     └── filter_and_score.py
├── nexa_train/
│     ├── train.py
│     ├── distill.py
│     └── configs/
│           └── science_student_v1.yaml
├── nexa_eval/
│     └── science_eval.py
├── scripts/
│     ├── run_data.sh
│     ├── run_train.sh
│     ├── run_eval.sh
│     └── run_sync.sh
├── runs/
│     └── manifests/
└── docs/
      └── CHANGELOG.md
```

---

## Environment Setup

Set the following environment variables before running pipeline stages (recommended in `.env` or bootstrap script):

```bash
export NEXA_SCRATCH=/workspace/tmp
export NEXA_DURABLE=/mnt/nexa_durable
export NEXA_SHARED=/workspace/shared
export WANDB_API_KEY=<your_key>
export HF_TOKEN=<your_key>
```

---

## Storage Hierarchy

| Tier       | Path                   | Usage                             |
|------------|------------------------|-----------------------------------|
| **tmp**    | /workspace/tmp         | Ephemeral artifacts (logs, chkpnt)|
| **perm**   | /mnt/nexa_durable      | S3-backed persistent data         |
| **shared** | /workspace/shared      | Shared cache (tokenizers, configs)|

---

## Scripts

All core operations are scripted for reproducibility and automation.  
**Remember:** Make all scripts executable: `chmod +x scripts/*.sh`

- **Data Preparation**
    - `scripts/run_data.sh`  
      ```bash
      #!/usr/bin/env bash
      set -e
      python -m nexa_data.make_distill_dataset \
        --input /mnt/nexa_durable/datasets/raw.parquet \
        --output /mnt/nexa_durable/datasets/pending.parquet
      ```
- **Training**
    - `scripts/run_train.sh`  
      ```bash
      #!/usr/bin/env bash
      set -e
      CONFIG=${1:-"nexa_train/configs/science_student_v1.yaml"}
      python -m nexa_train.distill --config $CONFIG
      ```
- **Evaluation**
    - `scripts/run_eval.sh`  
      ```bash
      #!/usr/bin/env bash
      set -e
      RUN_ID=${1:-latest}
      python -m nexa_eval.science_eval --run_id $RUN_ID
      ```
- **Sync to S3**
    - `scripts/run_sync.sh`  
      ```bash
      #!/usr/bin/env bash
      aws s3 sync /mnt/nexa_durable s3://nexacompute/ --exclude "*.tmp"
      ```

---

## Manifests & Logging

Every significant stage logs a manifest (JSON) in `/runs/manifests`.  
A canonical manifest schema example:

```json
{
  "run_id": "train_20251101_143200",
  "stage": "train",
  "status": "completed|failed|retry",
  "model": "mistral-7b",
  "dataset": "science_assistant_v1",
  "checkpoint": "/mnt/nexa_durable/checkpoints/train_20251101_143200/final.pt",
  "logs": "/mnt/nexa_durable/logs/train_20251101_143200.log",
  "eval_report": null,
  "hardware": "RTX 5090",
  "cost_usd": 8.50
}
```
All runs also log W&B run IDs for experiment tracking and cost/GPU hours as soon as possible.

---

## Pipeline Workflow

A standard pipeline run consists of:

1. **Data Preparation**
   ```bash
   bash scripts/run_data.sh
   ```
2. **Teacher Generation**
   ```bash
   python -m nexa_data.generate_from_teacher \
     --input /mnt/nexa_durable/datasets/pending.parquet \
     --output /mnt/nexa_durable/datasets/generated.parquet
   ```
3. **Filtering & Scoring**
   ```bash
   python -m nexa_data.filter_and_score \
     --input /mnt/nexa_durable/datasets/generated.parquet \
     --output /mnt/nexa_durable/datasets/science_assistant_v1.parquet
   ```
4. **Training**
   ```bash
   bash scripts/run_train.sh configs/science_student_v1.yaml
   ```
5. **Evaluation**
   ```bash
   bash scripts/run_eval.sh train_<run_id>
   ```
6. **Sync Outputs**
   ```bash
   bash scripts/run_sync.sh
   ```

---

## Configuration: Example Training YAML

```yaml
model_name: "mistral-7b-instruct"
dataset_path: "/mnt/nexa_durable/datasets/science_assistant_v1.parquet"
lr: 2e-5
batch_size: 1
grad_accumulation: 8
epochs: 2
use_kd: true
kd_temperature: 1.5
kd_lambda: 0.5
save_steps: 500
eval_steps: 500
fp16: true
```

---

## Evaluation Process

Evaluations are rubric-based and output to  
`/mnt/nexa_durable/evals/science_eval_<run_id>.parquet`

- **Criteria:**  
  - *Falsifiability*
  - *Novelty*
  - *Grounding*
  - *Completeness*
  - *Safety*

- **Scoring:**  
  ```
  DSS = 0.3*falsifiability + 0.2*grounding +
        0.2*completeness + 0.2*safety + 0.1*novelty
  ```

---

## Immediate TODO (Example Next Steps)

1. Implement Axolotl backend + configs (`nexa_train/backends/axolotl.py`).
2. Extend dataset tooling (`make_distill_dataset.py`) with validation + sampling policies.
3. Stand up CI pipeline for end-to-end chain execution (`pipeline.yml` + CLI/API).
4. Introduce dataset/version registry with checksums and schema validation.
5. Expand cost + telemetry dashboards (GPU hours, spend baselines).
6. Automate deployment promotion workflow (package → deploy → smoke test).
7. Snapshot node images/pipeline state for reproducibility.
8. Document changes and runbook updates for every release.

---

## Job Launch System: Modular, Layered Approach

### Layers

**1. Config:**  
- YAML/JSON configures the job and is never hard-coded.

**2. Runner:**  
- Slim Python module that loads config, executes, and writes a manifest.  
- Examples:
  - `nexa_compute/training/hf_runner.py` (standard HuggingFace fine-tune)
  - `nexa_train/backends/axolotl.py` (future: Axolotl/DeepSpeed)
  - `nexa_eval/eval_runner.py` (evaluations)

**3. Controller:**  
- Scripts that only invoke the runner with provided config.
  ```bash
  python -m nexa_train.launcher --config configs/train_sst2.yaml
  ```

---
### Job Lifecycle Table

| Stage     | Example                                           | Output                                   |
|-----------|---------------------------------------------------|------------------------------------------|
| **data**  | `bash scripts/run_data.sh`                        | parquet dataset                          |
| **train** | `python -m nexa_train.launcher --config configs/train_sst2.yaml` | /mnt/nexa_durable/checkpoints/{run} |
| **eval**  | `bash scripts/run_eval.sh train_20251101_xxxx`    | eval parquet                             |
| **sync**  | `bash scripts/run_sync.sh`                        | upload to S3                             |

---

### Design Rules

- All scripts < 300 lines.
- Each stage = one entrypoint.
- All jobs log a manifest & W&B run ID.
- Only `run_sync.sh` interacts with S3.
- Strict YAML/JSON configurations.

### Current Implementation Snapshot (Nov 2025)

- `nexa_train/launcher.py` is the canonical entrypoint: it resolves configs, dispatches to backends, and prints manifest locations.
- `nexa_train/backends/hf.py` maps launcher params into `HFTrainingConfig` and executes `nexa_compute/training/hf_runner.py`.
- `scripts/run_train_ddp.sh` wraps `torchrun` with sane defaults for RTX 4090/5090-class nodes (NCCL, WANDB, socket envs).
- `scripts/cleanup.sh` prunes scratch + durable checkpoints and archives aged runs for compliance.
- `docker/cuda_12.4.txt` pins Torch/Triton/XFormers for CUDA 12.4 images; use with UV/Docker builds.
- `configs/acc/fsdp_4gpu.yaml` provides an Accelerate FSDP profile for 4×GPU jobs (bf16, full shard, offload).
- `scripts/submit_slurm.py` + `nexa_infra/slurm.py` generate/submit Slurm arrays with cost estimation and job manifests.

---

## Distributed Training Specification

NexaCompute is fully equipped for scalable and reliable distributed (multi-GPU) training, supporting both standard and advanced parallelism techniques.

### Modes Supported

| Mode            | Use Case                      | Launcher                    | Backend |
|-----------------|------------------------------|-----------------------------|---------|
| DataParallel (DP)   | Quick 2–4 GPU runs           | `torchrun --nproc_per_node` | nccl    |
| DistributedDataParallel (DDP) | Large model, single node | `torch.distributed.run`     | nccl    |
| FSDP (future)   | 20B+ param models             | Axolotl/Deepspeed (planned) | nccl    |

> **Default:** Single-node multi-GPU DDP in v1.1

### Multi-GPU Launch Pattern

- Launch with:
  ```bash
  torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    -m nexa_train.entrypoint \
    --config configs/train_sst2.yaml
  ```
- Each GPU gets its rank; only rank 0 saves checkpoints, writes manifest.

### TrainingArguments Guidelines

```python
TrainingArguments(
    ddp_find_unused_parameters=False,
    ddp_backend="nccl",
    gradient_accumulation_steps=grad_accum,
    per_device_train_batch_size=batch_size,
    ...
)
# Optionally:
import torch.distributed as dist
dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
```

### Environment Configuration

Set these before launch (**example for bootstrap.sh**):

```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export WANDB_START_METHOD=thread
```
- Adjust `NCCL_SOCKET_IFNAME` to match network interface (`ip a` to check).

### Multi-GPU Launch Script Example

`scripts/run_train_ddp.sh`:

```bash
#!/usr/bin/env bash
set -e
CONFIG=${1:-"configs/train_sst2.yaml"}
GPUS=${2:-4}
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export CUDA_LAUNCH_BLOCKING=1
export WANDB_START_METHOD=thread

torchrun --nproc_per_node=$GPUS -m nexa_train.entrypoint --config $CONFIG
```
Usage:
```bash
bash scripts/run_train_ddp.sh configs/train_sst2.yaml 4
```

### Storage Layout During DDP

Every rank writes to its own scratch directory:
```
/workspace/tmp/train_<run_id>/rank_<rank_id>/
```
**Only rank 0** persists checkpoints, manifests, and uploads to S3.

### Key Sanity Checks Before & During Launch

- Verify GPUs:
  ```bash
  python -c "import torch; print(torch.cuda.device_count()); print(torch.distributed.is_available())"
  ```
- Monitor GPU utilization:  
  `watch -n 1 nvidia-smi`
- Check correct network interface and software/driver compatibility.

### Debugging Checklist

- GPUs visible in `nvidia-smi`
- Environment variables correct (esp. network interface)
- Start with 2-GPU tests
- Enable `NCCL_DEBUG=INFO`
- Log rank IDs in W&B for troubleshooting

### Common NCCL Issues & Solutions

| Symptom                         | Fix/Workaround                   |
|----------------------------------|----------------------------------|
| `NCCL WARN Watchdog` / Hang      | Upgrade Torch/CUDA/NCCL versions |
| `Error 2 (Internal error)`       | Set correct `NCCL_SOCKET_IFNAME` |
| `collectives failed`             | Reduce batch size, validate ranks|
| `timeout in allreduce`           | Raise timeout (to 30 minutes)    |
| Silent freeze (W&B bug)          | Use `WANDB_START_METHOD=thread`  |

---

## Operational Practices

- Launch jobs via scripts and tmux for reliability:
  ```bash
  tmux new -s train
  cd ~/nexa-compute
  bash scripts/run_train.sh configs/train_sst2.yaml
  ```
  Detach: `Ctrl+B, D`  |  Reattach: `tmux attach -t train`
- Store long-term model, dataset, artifact assets on S3 via `run_sync.sh`
- Never hard-code any runner's config

---

## Scaling Roadmap

| Version | Capability                                | Tools/Backends            |
|---------|-------------------------------------------|---------------------------|
| v1.1    | Reliable DDP (single-node)                | Torch DDP                 |
| v1.2    | Multi-node (cross-pod) DDP                | Tailscale + torchrun      |
| v1.3    | Sharded/gigaparameter support             | Axolotl/DeepSpeed         |
| v1.4    | Hybrid inference serving (CPU+GPU)        | VLLM / Triton             |

---

## Summary

- Unified, layered, and scriptable infrastructure for ML distillation with strong manifesting/logging.
- Rigid environment & storage conventions to promote reproducibility.
- Modular job launching via `nexa_train/launcher.py` + backend registry (HF live, Axolotl upcoming).
- Production-grade distributed training (torchrun wrapper, Accelerate FSDP profile) with single-node DDP by default.
- Slurm batch generation with cost tracking and S3/offline cleanup policies baked in.
- Scale-out path ready for multi-node and future Axolotl/DeepSpeed + inference serving.

Refer to this document when onboarding new contributors, establishing automated jobs, debugging distributed runs, or scaling to larger workloads. For implementation specifics, consult module READMEs and docstrings.

Perfect, let’s squeeze more juice out of this while you’re in builder mode. Here’s a **single technical markdown** you can drop into `docs/ROADMAP_OPS.md` — focused on smoothness, ops, and a path from “HF on 1 GPU” → “Axolotl on 8 GPUs” → “20–60B class” so you can look serious to paying clients.

---

# NexaCompute – Unified Infra: What This Stack Must Do

This section is the **source of truth** for the goals, design, and operational contracts of NexaCompute infrastructure. It encodes what we *need* this stack to deliver for current and future requirements—both for everyday reliability and for client-credible, scale-out science LLM work.

---

## 1. Runtimes & Environments

**Requirements:**

- Environment management must be fast, reliable, and reproducible across CI, cloud, and physical server.
- CUDA and PyTorch versions must always match host driver/GPU stack.
- No “works on my box” failures; onboarding should be zero pain for new contributors.

**Implementation:**

- Project uses [UV](https://github.com/astral-sh/uv) for Python env lock and reproducibility.
- All dependencies except CUDA/PyTorch in `uv.lock` and managed by UV in `.venv`; bootstrap script installs GPU deps to match system image.
    - Example:
      ```bash
      # scripts/bootstrap_env.sh
      uv venv .venv
      source .venv/bin/activate
      uv pip install -r requirements.txt
      uv pip install -e .
      # Then install torch as pinned for the node:
      pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
      ```
- Exact torch/pytorch/torchvision/torchaudio pins are tracked in `docker/cuda_12.4.txt` for infra reproducibility.

---

## 2. Trainer Abstraction Layer

The project must have:

- A **single idiomatic way** (“launcher interface”) to trigger training regardless of backend (HF, Axolotl, Accelerate).
- All config is **YAML/JSON**, absolutely no hardcoded experiment or runner params.
- Adding a new backend should only require:
    - Implementing a module in `nexa_train/backends/`
    - Registering it in the launcher.

**Contract:**
- All runners:
    - Accept a config path.
    - Save a manifest upon completion.
    - Respect CLI-overrides.

**Pattern:**
```text
nexa_train/
  backends/
    hf.py          # plain transformers
    axolotl.py     # LoRA/QLoRA/DS
    accelerate.py  # FSDP/sharded
  launcher.py      # single entrypoint, backend registry
```
Example code:
```python
# nexa_train/launcher.py
BACKENDS = {
  "hf": "nexa_train.backends.hf:run",
  "ax": "nexa_train.backends.axolotl:run",
  "acc": "nexa_train.backends.accelerate:run",
}
```
Example invocation:
```bash
python -m nexa_train.launcher --backend hf --config configs/train/science.yaml
python -m nexa_train.launcher --backend ax --config configs/axolotl/science_lora.yaml
```
**Result:** You can swap frameworks, scale up complexity, mix-and-match features, without changing your pipeline logic.

---

## 3. Axolotl Path for Advanced/Client Training

- Axolotl is the de facto backend for LoRA, QLoRA, DeepSpeed, FSDP, quantized fine-tuning, and efficient 4/8-bit runs.
- **To add:**
    - `nexa_train/backends/axolotl.py`: loads YAML, runs `axolotl.cli.main()`, captures output/manifest.
    - `scripts/run_axolotl.sh` for shell script invocation.
    - Durable storage is always mounted in `/mnt/nexa_durable`; checkpoint outputs go under `$RUN_ID`.

- **We guarantee:** Out of the box, you can support Axolotl and DeepSpeed, for 8x GPU, sharded, and quantized jobs, with no extra infra hacks.

---

## 4. Distributed DDP / Multi-GPU Reliability

- DDP is default for training. Multi-GPU must be plug-and-play.
- Implement these two env settings in every multi-GPU runner:
    ```bash
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    ```
- These should be set by default in any `run_train_ddp.sh`.

---

## 5. Memory & Scaling Strategies (20B+ Models)

- **Three codepaths must exist:**
    1. **Quantized path:** 4-bit (bnb), via Axolotl, for fastest fitting of big models on limited GPU.
    2. **Sharded path:** FSDP or ZeRO via Accelerate/DeepSpeed, for 20–60B.
        ```yaml
        mixed_precision: bf16
        fsdp: full_shard
        fsdp_offload_params: true
        fsdp_state_dict_type: FULL_STATE_DICT
        ```
        Example launch:
        ```bash
        accelerate launch --config_file configs/acc/fsdp_4gpu.yaml \
          -m nexa_train.launcher --backend acc --config configs/train/science.yaml
        ```
    3. **Offload path:** must allow optimizer & params to CPU/NVMe (`/workspace/tmp` as scratch if needed).

- **Rule:** If `model_size > 15B`, pipeline must enforce `axolotl` or `accelerate` backend; do not allow raw HF runner.

---

## 6. Scaling Matrix — What Infra Must Support

| Tier   | Model Size      | Hardware                 | Backend(s)                    |
| ------ | --------------- | ------------------------ | ----------------------------- |
| T1     | ≤ 13B           | 1×80GB or 2×48GB         | HF / DDP                      |
| T2     | 20–30B          | 4–8×48GB, NVLink ideal   | Axolotl (DS/FSDP)             |
| T3     | 40–60B          | 8×A100/H100 or multinode | Accelerate FSDP / DS ZeRO-3   |
| T4     | >60B (adapter)  | multinode, fast network  | LoRA/adapters only            |

**Narrative for clients:** We robustly support up to T3 training and T4 via adapters/LoRA.

---

## 7. Operations & Reliability (“Smoothness”)

### 7.1 Manifest tagging

- Every run manifest **must** have freeform tags, e.g.:
    ```json
    "tags": ["backend:ax", "train", "experiment:science_v1"]
    ```
  This allows for powerful audit/filter/search.

### 7.2 Automated Cleanup

- Provide and invoke `scripts/cleanup.sh` **at the end of every major pipeline step** to keep scratch clean and reduce S3 clutter.
    ```bash
    find /workspace/tmp -maxdepth 2 -mtime +1 -type d -exec rm -rf {} \;
    aws s3 ls s3://nexacompute/checkpoints/
    ```

### 7.3 Basic health check

- Ship `scripts/diag.sh`; use at pipeline start:
    ```bash
    nvidia-smi
    python -c "import torch; print(torch.cuda.device_count())"
    echo $NCCL_SOCKET_IFNAME
    ```

---

## 8. Multi-node / R&D (“NexaDist” Path) — Futureproofing

- **NexaDist** defines the path to true multinode scaling (60B+ ambitions).
    - Python controller reads `cluster.yaml` (host IPs, GPU counts)
    - Assigns ranks; launches `torchrun` via SSH (over Tailscale/WireGuard)
    - DDP/FSDP under the hood; basic rdzv + restarts
    - Example config:
      ```yaml
      cluster:
        - host: 100.66.164.18
          gpus: 4
        - host: 100.66.164.19
          gpus: 4
      backend: torch.distributed
      rdzv: 100.66.164.18:29500
      ```
      Example run:
      ```bash
      python -m nexa_dist.launch --config configs/dist/8gpu.yaml
      ```
    - You do not have to build this now, but must have a stub and a plan.

---

## 9. Inference & Model Serving (Productize Loop)

- Add `nexa_infer/vllm_runner.py` for inference; uses vLLM, loads checkpoints from `/mnt/nexa_durable/checkpoints/<run_id>/final`.
- Add `scripts/run_vllm.sh` for easy server launch.

    Example:
    ```bash
    python -m nexa_infer.vllm_runner --model /mnt/nexa_durable/checkpoints/...
    ```

**Must** be possible to train → eval → serve on same node or VM.

---

## 10. Essential TODOs to Hit Table Stakes

- [x] Implement modular backend launcher (`nexa_train/launcher.py`)
- [x] Implement minimal `nexa_train/backends/hf.py`
- [x] Provide DDP script `scripts/run_train_ddp.sh`
- [x] Provide scratch janitor `scripts/cleanup.sh`
- [x] Provide docker pin file `docker/cuda_12.4.txt`
- [x] Provide FSDP config example `configs/acc/fsdp_4gpu.yaml`
- [ ] Implement Axolotl backend (`nexa_train/backends/axolotl.py`) with configs
- [ ] Ship Accelerate/FSDP benchmarking suite and automated regression tests
- [ ] Build inference runner (`nexa_infer/vllm_runner.py`) + `scripts/run_vllm.sh`

**Summary:**
> If you want a system that supports HF, Axolotl, DDP, and a clean story toward FSDP/60B, this is *the* contract and architectural baseline to deliver on.

---
