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

## Full Documentation

For complete details, see the other documentation files in this directory.
