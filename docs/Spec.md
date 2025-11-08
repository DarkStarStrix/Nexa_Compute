---
title: System Specification
slug: reference/spec
description: Authoritative reference for the NexaCompute system design.
---

# **NexaCompute v1 — Full System Specification**

**Author:** Allan
**Date:** October 28, 2025
**Version:** v1.0 (Production Spec)
**Status:** Implementable

---

## **0. Purpose**

**NexaCompute** is a self-contained, production-grade applied ML R&D lab that runs on ephemeral GPU compute across multiple providers (Lambda Labs, Prime Intellect, CoreWeave, RunPod, etc.).

It enables:

* Rapid fine-tuning and distillation of high-value domain models (scientific reasoning, experimental planning, materials discovery, etc.).
* Rigorous evaluation against baselines and teacher models.
* Automated data-driven feedback loops.
* Reproducible, cost-aware, and safe experimentation at scale.

**Core loop:**

1. Generate or curate aligned datasets.
2. Train or distill models.
3. Evaluate them on structured tasks.
4. Use evaluation gaps to improve data.
5. Repeat.

Everything runs on disposable compute, with durable results and lineage recorded permanently.

---

## **1. System Overview**

NexaCompute is structured as a **modular monolith** consisting of six layers:

| Layer | Module           | Role                                                               |
| ----- | ---------------- | ------------------------------------------------------------------ |
| L0    | `nexa_infra/`    | Infrastructure provisioning, Tailscale network, and orchestration. |
| L1    | `nexa_data/`     | Data preparation, augmentation, schema validation, and versioning. |
| L2    | `nexa_train/`    | Model fine-tuning, distillation, and hyperparameter sweeps.        |
| L3    | `nexa_eval/`     | Evaluation and benchmarking engine (LLM-as-judge + metrics).       |
| L4    | `nexa_feedback/` | Iterative feedback loop to improve data based on eval signals.     |
| L5    | `nexa_ui/`       | Visualization and delivery (leaderboards, dashboards).             |

The repository is designed to maintain **clean separation via artifacts**, not imports — each layer consumes and produces versioned outputs (datasets, checkpoints, manifests, eval reports).

---

## **2. Repository Structure**

```text
nexa-compute/
├── orchestrate.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
│
├── nexa_infra/
│   ├── provision.py
│   ├── sync_code.py
│   ├── launch_job.py
│   ├── teardown.py
│   ├── cost_tracker.py
│   ├── cluster.yaml
│   ├── tailscale_bootstrap.sh
│   └── utils.py
│
├── nexa_data/
│   ├── prepare.py
│   ├── augment.py
│   ├── distill_materialize.py
│   ├── filters/
│   ├── schemas/
│   ├── loaders/
│   └── manifest/dataset_registry.yaml
│
├── nexa_train/
│   ├── train.py
│   ├── distill.py
│   ├── configs/
│   ├── sweeps/
│   ├── optim/
│   ├── models/
│   └── utils.py
│
├── nexa_eval/
│   ├── generate.py
│   ├── judge.py
│   ├── analyze.py
│   ├── rubrics/
│   ├── tasks/
│   └── reports/
│
├── nexa_feedback/
│   ├── feedback_loop.py
│   ├── weakness_analysis.py
│   └── feedback_generators/
│
├── nexa_ui/
│   ├── leaderboard.py
│   ├── dashboards/
│   └── static/
│
├── runs/
│   ├── manifests/
│   ├── logs/
│   └── checkpoints/
│
└── docs/
    ├── ARCHITECTURE_v1.md
    ├── STORAGE_POLICY.md
    ├── RUN_MANIFEST_SCHEMA.md
    ├── DATA_FORMAT.md
    ├── EVAL_FRAMEWORK.md
    ├── COST_MODEL.md
    └── SAFETY_POLICY.md
```

---

## **3. Infrastructure Layer (L0) — `nexa_infra/`**

### **Purpose**

Spin up, access, and tear down GPU nodes across providers using Tailscale and SSH.
Compute is ephemeral; results are durable.

### **Key Scripts**

#### `provision.py`

* Allocates GPU instance from provider API (Lambda/CoreWeave/Prime Intellect).
* Installs Docker, Python, Tailscale (via `tailscale_bootstrap.sh`).
* Uses pre-auth key to auto-join tailnet:

  ```bash
  tailscale up --authkey ${TAILSCALE_KEY} --hostname nexa-${RUN_ID} --ssh
  ```
* Mounts `/mnt/nexa_durable` (cloud bucket or persistent volume).
* Returns:

  * hostname
  * tailscale_ip
  * ssh_user
  * gpu_type
  * scratch_path

#### `sync_code.py`

* Rsyncs repo to `/workspace/nexa-compute` on remote node via Tailscale:

  ```bash
  rsync -avz --exclude-from='.gitignore' . ubuntu@${TAILSCALE_HOST}:/workspace/nexa-compute
  ```

#### `launch_job.py`

* SSH via Tailscale and execute training/eval job:

  ```bash
  ssh ubuntu@${TAILSCALE_HOST} "cd /workspace/nexa-compute && ./scripts/shell/training/run_training.sh"
  ```
* Sets environment:

  * `NEXA_SCRATCH=/scratch`
  * `NEXA_DURABLE=/mnt/nexa_durable`
  * `NEXA_RUN_ID=<timestamp>`
  * `NEXA_CHECKPOINT_DIR=/scratch/checkpoints/$RUN_ID`
  * `NEXA_DURABLE_CHECKPOINT_DIR=/mnt/nexa_durable/checkpoints/$RUN_ID`

#### `teardown.py`

* Syncs checkpoints/logs from scratch → durable.
* Writes final manifest to `/mnt/nexa_durable/manifests/`.
* Destroys node via provider API.

#### `cost_tracker.py`

* Records:

  * `start_time`, `end_time`
  * `gpu_type`, `provider`
  * `$ cost/hour`, `$ total`
* Updates run manifest accordingly.

---

## **4. Data Engine (L1) — `nexa_data/`**

### **Purpose**

Produce versioned, schema-validated datasets that drive training and distillation.

### **Functions**

* `prepare.py` — cleans and formats dataset, writes Parquet to `/mnt/nexa_durable/datasets/`.
* `augment.py` — synthetic sample generation (teacher prompting).
* `filters/` — safety and quality filters (deduplication, unphysical flagging).
* `schemas/` — JSON schemas for each dataset type.
* `manifest/dataset_registry.yaml` — maps dataset IDs → paths, hashes, notes.

### **Schema Example**

```json
{
  "context": "string",
  "prompt": "string",
  "target_answer": {
    "hypothesis": "string",
    "methodology": "string",
    "risks": "string"
  },
  "domain": "string",
  "source": "human|synthetic",
  "quality_score": "float"
}
```

---

## **5. Training & Distillation Engine (L2) — `nexa_train/`**

### **Purpose**

Convert datasets into fine-tuned or distilled models.

### **Modes**

1. **Supervised fine-tuning:** SFT/IFT on curated data.
2. **Distillation:** teacher (e.g. GPT-4/Claude) → student (smaller open model).

### **Key Scripts**

* `train.py` — runs training with config and logs to W&B.
* `distill.py` — queries teacher API, materializes paired data, trains student.
* `sweeps/launch_sweep.py` — runs multiple configs via W&B sweeps.
* `sweeps/analyze_sweep.py` — selects best config → updates `configs/autotuned.yaml`.
* `optim/fsdp_wrapper.py` — FSDP/tensor-parallel harness.

### **Artifacts**

* Checkpoints → `/scratch/checkpoints/<run_id>/`
* Synced to → `/mnt/nexa_durable/checkpoints/<run_id>/`
* Manifest → `/mnt/nexa_durable/manifests/<run_id>.json`

---

## **6. Evaluation Engine (L3) — `nexa_eval/`**

### **Purpose**

Benchmark model performance, cost, and safety.

### **Flow**

1. `generate.py` — runs each model on eval prompts.
2. `judge.py` — scores results using LLM-as-judge + heuristics.
3. `analyze.py` — builds leaderboard and aggregates metrics.

### **Metrics**

* Falsifiability
* Clarity
* Experimental design quality
* Risk awareness
* Safety compliance
* Latency / Cost per prompt

### **Rubric Hashing**

Each evaluation prompt has:

* `judge_model_id`
* `judge_prompt_hash`
  Recorded in leaderboard and run manifests for reproducibility.

---

## **7. Feedback Loop (L4) — `nexa_feedback/`**

### **Purpose**

Automate improvement by targeting weak areas from evaluation.

### **Steps**

1. `weakness_analysis.py` finds low-scoring domains (e.g., “falsifiability_score < 4.0”).
2. `feedback_generators/` produce new data for weak domains (via teacher prompting).
3. `feedback_loop.py` merges new data, creates new dataset version, updates registry.
4. Next run retrains/distills on improved data.

---

## **8. Delivery & UI (L5) — `nexa_ui/`**

### **Purpose**

Display progress and publish results.

### **Modules**

* `leaderboard.py` — Streamlit/Flask dashboard showing model vs teacher vs baselines.
* `dashboards/` — visualizations for:

  * cost vs quality
  * domain performance
  * improvement over time
* Export static HTML/PDF reports for shipping or portfolio inclusion.

---

## **9. Storage & Artifact Governance**

| Tier                     | Path                           | Purpose                                           | Retention                        |
| ------------------------ | ------------------------------ | ------------------------------------------------- | -------------------------------- |
| **A. Ephemeral Scratch** | `/scratch`                     | Fast local storage for training                   | Destroy after run                |
| **B. Durable Storage**   | `/mnt/nexa_durable`            | Permanent bucket for datasets, checkpoints, evals | Keep canonical results           |
| **C. Local Ledger**      | `runs/`, `nexa_data/manifest/` | Versioned in git                                  | Permanent (small JSON/YAML only) |

---

### `.gitignore` Excerpt

```gitignore
# Artifacts and large data
runs/checkpoints/
nexa_data/artifacts/
**/*.pt
**/*.parquet
**/*.jsonl
wandb/
scratch/
mnt/nexa_durable/
```

**Rule:** Git stores *metadata and manifests only*.
No large binaries, no datasets, no checkpoints.

---

## **10. Run Manifest Schema**

Each run generates a manifest stored at `/mnt/nexa_durable/manifests/<run_id>.json` and synced locally.

```json
{
  "run_id": "2025-10-28_141200",
  "run_type": "distill",
  "dataset_id": "materials_hypothesis_v2",
  "dataset_hash": "ab12cd34",
  "teacher_model": "gpt-4o",
  "student_model": "mistral-7b-lora",
  "gpu_type": "A100x4",
  "provider": "Lambda",
  "tailscale_ip": "100.64.23.45",
  "hostname": "nexa-2025-10-28_141200",
  "wandb_run_id": "nexa-abc123",
  "tokens_trained": 480000000,
  "cost_estimate_usd_total": 22.45,
  "checkpoint_dir_durable": "/mnt/nexa_durable/checkpoints/2025-10-28_141200/",
  "best_checkpoint": "final.pt",
  "eval_status": "done",
  "eval_report": "/mnt/nexa_durable/evals/reports/leaderboard_2025-10-28.parquet",
  "eval_summary": {
    "falsifiability": 4.5,
    "clarity": 4.3,
    "safety": 5.0
  }
}
```

---

## **11. Tailscale Integration**

### **Setup**

1. Create a Tailscale account and tailnet.
2. Generate pre-auth keys (`tailscale admin → Keys → Pre-auth key`).
3. Store in `.env`:

   ```bash
   TAILSCALE_AUTH_KEY=tskey-abc123
   ```

### **Bootstrap Script (`tailscale_bootstrap.sh`)**

```bash
#!/bin/bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey ${TAILSCALE_AUTH_KEY} --hostname nexa-${RUN_ID} --ssh --accept-dns
```

### **Usage**

* `provision.py` runs bootstrap after node creation.
* All SSH and rsync commands use Tailscale IP/hostname:

  ```bash
  ssh ubuntu@${TAILSCALE_HOST}.tailnet "ls /workspace"
  ```

---

## **12. Weights & Biases Integration**

* Every `train.py` / `distill.py` run initializes:

  ```python
  wandb.init(project="nexa-compute", name=run_id)
  ```
* Logs metrics, losses, throughput, GPU memory, etc.
* `sweeps/launch_sweep.py` triggers multi-run sweeps via W&B.
* `sweeps/analyze_sweep.py` pulls best results → updates `autotuned.yaml`.

---

## **13. Safety & Compliance**

### **Technical Safety**

* Filters flag unphysical or unsafe experiment proposals.
* Scoring rubric includes `safety_score` and `red_flag_count`.
* Unsafe generations are excluded from next dataset versions.

### **Human Review**

* Before public release or client delivery:

  * Manually review random samples in `nexa_ui/leaderboard.py`.
  * Exclude hallucinated or risky results.

---

## **14. Operating Principles**

1. **If it’s not logged, it didn’t happen.**
2. **Never rely on ephemeral storage.**
3. **No eval = not real.**
4. **Data is first-class.**
5. **Always measure cost vs quality.**
6. **Safety before publication.**

---

## **15. Quickstart Example**

```bash
# 1. Generate dataset
python orchestrate.py --stage data_gen --dataset plasma_v1

# 2. Train model
python orchestrate.py --stage train --mode distill --teacher gpt-4o --student mistral-7b

# 3. Evaluate
python orchestrate.py --stage eval --run_id 2025-10-28_141200

# 4. Feedback loop
python orchestrate.py --stage feedback
```

Artifacts and lineage are automatically tracked.

---

## **16. Future Extensions**

* `nexa_rl/` — Reinforcement learning or active learning.
* `nexa_monitor/` — Node telemetry + GPU utilization dashboards.
* `nexa_registry/` — Model registry for versioned deployments.
* Integration with Exa.ai, Hugging Face Hub, or MLflow.

---

## **17. Compliance & Cost Management**

* Ephemeral nodes are auto-terminated after inactivity.
* `cost_tracker.py` logs provider, runtime, cost per run.
* Datasets and checkpoints are hashed and versioned for audit.
* Sensitive data filtered via `filters/safety_filter.py`.

---

## **18. Deliverables**

Each successful experiment yields:

1. **Model checkpoint** (student/distilled).
2. **Run manifest** (JSON).
3. **Leaderboard entry** (Parquet).
4. **Evaluation report** (summarized).
5. **Cost breakdown** (logged).
6. **Data lineage proof** (registry entries).

Together, these form a complete scientific and engineering artifact.

---

### **End of Document**
