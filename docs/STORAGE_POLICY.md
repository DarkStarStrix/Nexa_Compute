Here’s the clean **storage layout** for your NexaCompute setup — this is the mental and directory map you’ll want to follow across every machine:

---

## **1. Ephemeral Storage (`/workspace/tmp` or `/scratch`)**

* **Purpose:** Fast, temporary working area for live training and data loading.
* **Location:** On the GPU node’s local NVMe or ephemeral disk.
* **Lifecycle:** Dies when the node is torn down — *never assume persistence.*

**Contents:**

```
/workspace/tmp/
  ├── dataloader_cache/
  ├── checkpoints_temp/
  ├── logs_temp/
  └── wandb_offline/
```

**Rules:**

* Always write intermediate checkpoints and temporary logs here.
* Sync only **final artifacts** to permanent storage.
* Use fast I/O here to maximize GPU throughput.

---

## **2. Permanent / Durable Storage (`/mnt/nexa_durable`)**

* **Purpose:** Long-term storage for model outputs, datasets, manifests, evals.
* **Location:** Can be:

  * A mounted S3/Backblaze bucket, or
  * A persistent volume (on Prime Intellect or Lambda), or
  * A local directory on your Mac, synced via rsync.
* **Lifecycle:** Survives reboots and pod teardown.

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

* Treat this as **source of truth** for all artifacts.
* Rsync this directory to your Mac for archival:

  ```bash
  rsync -avz gpu-node:/mnt/nexa_durable/ ~/nexa_compute/durable/
  ```
* Never commit it to git — only track manifests and hashes.

---

## **3. Shared Storage (`/workspace/shared` or `/mnt/shared`)**

* **Purpose:** Collaboration and multi-node coordination.
* **Location:** Optional — could be an NFS mount or a Tailscale-shared directory.
* **Lifecycle:** Persistent but shared across multiple GPU nodes.

**Contents:**

```
/workspace/shared/
  ├── common_datasets/
  ├── eval_prompts/
  └── active_jobs/
```

**Rules:**

* Use for shared datasets or coordination files between jobs.
* Don’t use for high-speed training I/O; this is for metadata and coordination.

---

## **Summary Table**

| Type                    | Path                      | Persistence | Typical Contents                          | Notes                        |
| ----------------------- | ------------------------- | ----------- | ----------------------------------------- | ---------------------------- |
| **Ephemeral (Scratch)** | `/workspace/tmp`          | ❌           | Temp checkpoints, dataloader shards, logs | Fast local NVMe              |
| **Durable**             | `/mnt/nexa_durable`       | ✅           | Checkpoints, datasets, evals, manifests   | Synced back to Mac           |
| **Shared**              | `/workspace/shared`       | ✅           | Common datasets, coordination files       | Multi-node access            |
| **Local Control (Mac)** | `~/nexa_compute/durable/` | ✅           | Mirrored durable storage                  | Local backup + git manifests |

---

Once this structure is in place:

* `/workspace/tmp` → for **speed**
* `/mnt/nexa_durable` → for **truth**
* `/workspace/shared` → for **teamwork / orchestration**
* Your **Mac** → for **control plane, code, and replication**

Would you like me to add a `setup_storage.sh` script that auto-creates and exports these paths with the right permissions on any node (to match your bootstrap.sh)?
