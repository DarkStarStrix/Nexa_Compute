# NexaCompute Infrastructure – Final Runbook

This repository now contains a reproducible GPU training stack with:

- Hardened bootstrap script for fresh GPU nodes (`nexa_infra/Boostrap.sh`).
- Configurable Hugging Face trainer with telemetry, smoothing, checkpoint packaging, and AWS S3 backup (`scripts/test_hf_train.py`).
- Helper scripts for GPU monitoring, torchrun launches, deployment promotion, run analysis, and artifact cleanup.
- Infrastructure summary command (`orchestrate.py summary`) to capture environment fingerprints.

The sections below describe how to bring up a new node, run long jobs, ship artifacts, and shut everything down cleanly.

---

## 1. Bootstrap a Fresh GPU Node

1. SSH into the machine as root.
2. Copy `nexa_infra/Boostrap.sh` and execute:
   ```bash
   bash nexa_infra/Boostrap.sh
   ```
   This installs base packages, Tailscale, AWS CLI, pynvml, environment exports, and creates durable storage paths.
3. Log out and back in (or `source ~/.bashrc`) so the new environment variables apply, including:
   - `NEXA_SCRATCH`, `NEXA_DURABLE`, `NEXA_SHARED`, `NEXA_REPO`
   - NCCL defaults (`NCCL_DEBUG=INFO`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=0`)
   - Tokenizer/OMP envs (`TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=8`)
   - Default S3 destination `NEXA_S3_PREFIX=s3://nexacompute/ML_Checkpoints`

> **AWS Credentials**: Configure `aws configure` or attach an IAM role so `aws s3 sync` can upload checkpoints.

---

## 2. Run a Training Job

All jobs use the configurable runner:

```bash
./scripts/run_training.sh \
  --model roberta-base \
  --dataset glue --dataset-config sst2 \
  --train-samples 20000 --eval-samples 5000 \
  --batch-size 16 --grad-accumulation 2 \
  --epochs 4 --learning-rate 1e-5 \
  --allow-tf32 --telemetry-interval 5 \
  --tags infra-stable
```

Highlights:

- Exports `WANDB_API_KEY`, `PYTHONPATH=/workspace/nexa_compute/src`, `CUDA_VISIBLE_DEVICES` (default 0), and NCCL envs.
- Supports additional flags (`--fp16`, `--bf16`, `--s3-uri`, logging intervals, etc.).
- On completion it writes a manifest (`runs/manifests/<run_id>.json`), syncs the best checkpoint to durable storage, prunes old checkpoints, and mirrors `final/` to S3 (`s3://nexacompute/ML_Checkpoints/<run_id>`).

For multi-GPU or H100 jobs, use the torchrun wrapper:

```bash
./scripts/torchrun_wrapper.sh scripts/test_hf_train.py [args...]
```

---

## 3. Telemetry & Monitoring

- **GPU Utilisation**: run `./scripts/gpu_monitor.py --interval 5` in another tmux pane.
- **In-run metrics**: the training script logs smoothed loss, raw loss, GPU memory, and W&B run IDs.
- **NVML Snapshots**: manifests capture pre/post GPU utilisation.

---

## 4. Packaging, Deployment, Cleanup

- **Package for deployment**:
  ```bash
  python3 scripts/package_for_deployment.py <run_id>
  ```
- **Promote to deploy mount (e.g. `/mnt/nexa_durable/deploy/current_model`)**:
  ```bash
  python3 scripts/deploy.py <run_id>
  ```
- **Summarise runs** (average loss/runtime, etc.):
  ```bash
  python3 scripts/analyze_runs.py
  ```
- **Prune/archive old checkpoints**:
  ```bash
  python3 scripts/cleanup.py --days 7 --prune-size +2G
  ```

---

## 5. Reproducibility Snapshot

After any meaningful run, capture the system fingerprint:

```bash
python3 orchestrate.py summary
cat runs/manifests/infra_report.json
```

This records Python/Torch/CUDA versions, NCCL envs, GPU inventory, and existing manifests. Include it with experiment notes.

---

## 6. Shutdown Workflow

1. Cancel tmux training (`tmux send-keys -t train C-c`).
2. Run `python3 orchestrate.py summary` and `python3 scripts/cleanup.py` if desired.
3. Package/deploy any final checkpoints.
4. `aws s3 ls s3://nexacompute/ML_Checkpoints/` to verify uploads.
5. Shut down node (provider-specific, or `shutdown -h now`).

---

### Directory Overview

| Path | Purpose |
|------|---------|
| `nexa_infra/Boostrap.sh` | Node bootstrap script |
| `scripts/test_hf_train.py` | Main training runner |
| `scripts/run_training.sh` | Helper wrapper for env + runner |
| `scripts/gpu_monitor.py` | NVML telemetry |
| `scripts/package_for_deployment.py` | Bundle checkpoints/manifests |
| `scripts/deploy.py` | Promote packaged runs |
| `scripts/analyze_runs.py` | Manifests analytics |
| `scripts/cleanup.py` | Prune/archive checkpoints |
| `runs/manifests/` | Per-run manifests + infra summaries |
| `docs/` | Post-mortems, storage policy, cost model, etc. |

---

Tag this state as `infra-stable-v1` once satisfied. Future work can iterate on functionality (multi-node scheduling, advanced telemetry, cost dashboards) without revisiting scaffolding.
