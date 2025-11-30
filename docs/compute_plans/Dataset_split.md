# Nexa Dataset Manifest Builder

This document describes the **train/val/test split strategy** for large-scale spectrum datasets and provides a **single, production-ready utility script** for generating the dataset manifest after shard creation.

It is designed for the Nexa data pipeline:

```
HF Raw HDF5 → AstroData VM → Shard Builder → Manifest → Wasabi → H100 Training
```

---

# 📌 Train/Val/Test Split

For a dataset of **~550 million spectra**, the recommended split for self-supervised pretraining is:

* **Train: 98%**
* **Validation: 1%**
* **Test: 1%**

### Rationale

* SSL requires **maximum data density**.
* Even 1% is millions of spectra—more than sufficient for validation/testing.
* Consistent with large pretraining corpora.
* Ensures downstream reproducibility.

This split is automatically computed by the script below.

---

# 📌 Manifest Builder Script

Save this as:

```
nexa_data/build_manifest.py
```

This script:

* Discovers all `.parquet` Arrow shards
* Randomly shuffles using a global seed
* Computes a **98/1/1** split
* Optionally computes SHA256 checksums
* Writes a complete `dataset_manifest.json`
* Works with any future dataset version (v1, v2, v3...)

---

# 🚀 `build_manifest.py`

```python
#!/usr/bin/env python3
"""
NEXA DATASET MANIFEST BUILDER
-----------------------------
Discovers Arrow shards, computes deterministic train/val/test splits,
computes optional checksums, and writes a dataset manifest.

Usage:
    python build_manifest.py \
        --shards_dir /nvme/shards \
        --output_dir /nvme/manifests \
        --version gems_v1 \
        --checksum

This script is designed as a single, contained utility to keep your
Nexa data pipeline dead simple and fully reproducible.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------
# Utility: Compute SHA256
# -------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

# -------------------------------------------------------
# Main routine
# -------------------------------------------------------
def build_manifest(shards_dir, output_dir, version, seed, shard_size_bytes, checksum):

    shards_dir = Path(shards_dir)
    output_dir = Path(output_dir)

    # Find all parquet files
    shard_files = sorted([p for p in shards_dir.rglob("*.parquet")])

    if not shard_files:
        raise RuntimeError(f"No .parquet shard files found under {shards_dir}")

    print(f"Found {len(shard_files)} shards.")

    # Convert to filenames
    shard_names = [p.name for p in shard_files]

    # Deterministic shuffling
    random.seed(seed)
    random.shuffle(shard_names)

    n = len(shard_names)
    n_train = int(n * 0.98)
    n_val = int(n * 0.01)
    n_test = n - n_train - n_val

    train_split = shard_names[:n_train]
    val_split = shard_names[n_train:n_train + n_val]
    test_split = shard_names[n_train + n_val:]

    print(f"Split: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test.")

    # Optional checksum computation
    checksums = {}
    if checksum:
        print("Computing checksums...")
        for shard_path in shard_files:
            checksums[shard_path.name] = sha256_file(shard_path)

    # Manifest metadata
    manifest = {
        "version": version,
        "global_seed": seed,
        "shard_size_bytes": shard_size_bytes,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "splits": {
            "train": train_split,
            "val": val_split,
            "test": test_split
        },
        "checksums": checksums
    }

    # Write manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dataset_manifest.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {out_path}")
    print("\nDone.")

# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--shard_size_bytes", type=int, default=2147483648)
    ap.add_argument("--checksum", action="store_true")
    args = ap.parse_args()

    build_manifest(
        args.shards_dir,
        args.output_dir,
        args.version,
        args.seed,
        args.shard_size_bytes,
        args.checksum
    )
```

---

# 📌 Example Usage (AstroData VM)

```bash
python nexa_data/build_manifest.py \
  --shards_dir /nvme/shards \
  --output_dir /nvme/manifests \
  --version gems_v1 \
  --checksum
```

Then upload:

```bash
rclone copy /nvme/manifests/dataset_manifest.json \
  wasabi:nexa-ms/processed/gems_v1/manifests/
```

---

# 📌 What This Gives You

* Deterministic train/val/test splitting (98/1/1)
* Consistent dataset versions (`gems_v1`, `gems_v2`, ...)
* Optional SHA256 for integrity checks
* Fully standalone utility — no scattered helpers
* Tight integration with Wasabi storage + H100 pipeline

This concludes the dataset manifest system for Nexa. The next step is running this inside your AstroData preprocessing workflow once the shards are generated.
