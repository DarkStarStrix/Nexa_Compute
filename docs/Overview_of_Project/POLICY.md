---
title: Policies
slug: /overview/policies
description: Centralized storage, safety, and cost policies for NexaCompute workflows.
---

# NexaCompute Policies

This document consolidates safety, storage, and cost policies for the NexaCompute platform.

## Storage Policy

### Storage Hierarchy

NexaCompute uses a three-tier storage architecture for optimal performance and persistence.

#### 1. Ephemeral Storage (`/workspace/tmp` or `/scratch`)

**Purpose:** Fast, temporary working area for live training and data loading.

**Location:** On the GPU node's local NVMe or ephemeral disk.

**Lifecycle:** Dies when the node is torn down — *never assume persistence.*

**Contents:**
```
/workspace/tmp/
  ├── dataloader_cache/
  ├── checkpoints_temp/
  ├── logs_temp/
  └── wandb_offline/
```

**Rules:**
- Always write intermediate checkpoints and temporary logs here.
- Sync only **final artifacts** to permanent storage.
- Use fast I/O here to maximize GPU throughput.

#### 2. Permanent / Durable Storage (`/mnt/nexa_durable`)

**Purpose:** Long-term storage for model outputs, datasets, manifests, evals.

**Location:** Can be a mounted S3/Backblaze bucket, persistent volume, or local directory synced via rsync.

**Lifecycle:** Survives reboots and pod teardown.

**Contents:**
```
/mnt/nexa_durable/
  ├── datasets/
  │   ├── dataset_v1.parquet
  │   ├── dataset_v2.parquet
  ├── checkpoints/
  │   ├── run_20251030_213000/
  │   │   ├── final.pt
  │   │   ├── config.yaml
  │   │   └── logs.json
  ├── evals/
  │   ├── reports/
  │   │   ├── leaderboard_20251030.parquet
  │   └── outputs/
  └── manifests/
      ├── run_20251030_213000.json
      └── dataset_registry.yaml
```

**Rules:**
- Treat this as **source of truth** for all artifacts.
- Rsync this directory to local machine for archival:
  ```bash
  rsync -avz gpu-node:/mnt/nexa_durable/ ~/nexa_compute/durable/
  ```
- Never commit it to git — only track manifests and hashes.

#### 3. Shared Storage (`/workspace/shared` or `/mnt/shared`)

**Purpose:** Collaboration and multi-node coordination.

**Location:** Optional — could be an NFS mount or a Tailscale-shared directory.

**Lifecycle:** Persistent but shared across multiple GPU nodes.

**Contents:**
```
/workspace/shared/
  ├── common_datasets/
  ├── eval_prompts/
  └── active_jobs/
```

**Rules:**
- Use for shared datasets or coordination files between jobs.
- Don't use for high-speed training I/O; this is for metadata and coordination.

#### Summary Table

| Type                    | Path                      | Persistence | Typical Contents                          | Notes                        |
| ----------------------- | ------------------------- | ----------- | ----------------------------------------- | ---------------------------- |
| **Ephemeral (Scratch)** | `/workspace/tmp`          | ❌           | Temp checkpoints, dataloader shards, logs | Fast local NVMe              |
| **Durable**             | `/mnt/nexa_durable`       | ✅           | Checkpoints, datasets, evals, manifests   | Synced back to Mac           |
| **Shared**              | `/workspace/shared`       | ✅           | Common datasets, coordination files       | Multi-node access            |
| **Local Control (Mac)** | `~/nexa_compute/durable/` | ✅           | Mirrored durable storage                  | Local backup + git manifests |

### Local Development Storage

For local development, use the `data/` directory structure:

```
data/
├── raw/              # Raw input data (gitignored)
├── processed/         # Organized processed outputs
│   ├── distillation/
│   ├── training/
│   ├── evaluation/
│   └── raw_summary/
```

See `DATA_ORGANIZATION.md` in the project root for detailed organization.

---

## Safety Policy

### Model Safety

- **Dataset Lineage:** Enforce dataset lineage: manifests must include source provenance before promotion.
- **Evaluation Rubrics:** Maintain evaluation rubrics for bias, toxicity, and hallucination checks.
- **Human Review:** Require human-in-the-loop sign-off for any model destined for production surfaces.

### Operational Safety

- **Access Control:** Access to infrastructure scripts gated via IAM & audited.
- **Secrets Management:** Secrets managed through environment variables and not committed to source control.
- **Distributed Validation:** Distributed launches validate node health before joining the training job to avoid partial failure.

### Incident Response

- **Automatic Halt:** On evaluation failure or key metric regression, automatically halt deployment pipelines.
- **Post-Mortem:** Archive logs/checkpoints for post-mortem; run `nexa_feedback` workflow to capture remediation steps.

---

## Cost Policy

### Cost Tracking

The cost tracker aggregates run-level compute spend captured per resource type.

**Inputs:**
- `compute_hours`: GPU/CPU billing hours multiplied by instance hourly rate.
- `storage_gb_month`: Persistent volume usage normalised to monthly cost.
- `egress_gb`: Network data egress charges.

**Outputs:**

Generated manifests (`runs/manifests/cost_<run_id>.json`) record:

```json
{
  "run_id": "baseline-2025-01-01",
  "breakdown": {
    "compute_hours": 42.5,
    "storage_gb_month": 12.4,
    "egress_gb": 1.2
  },
  "total": 56.1
}
```

**Aggregation:** `summarize_costs` aggregates totals for rapid reporting. For production usage integrate with billing APIs (AWS Cost Explorer, GCP Billing) and feed into this structure.

### Cost Optimization

- Use ephemeral storage for temporary files to minimize persistent volume costs.
- Archive completed runs to cold storage after retention period.
- Monitor egress charges and prefer local storage when possible.

