#!/usr/bin/env python3
"""Package a trained run into a deployment-ready bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from nexa_compute.utils.storage import get_storage  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a NexaCompute run for deployment")
    parser.add_argument("run_id", type=str, help="Run identifier (e.g. train_20251030_233000)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/deployment"), help="Local packaging dir")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing package")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage = get_storage()

    manifest_path = storage.manifest_path(args.run_id)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    checkpoint_dir = Path(manifest["checkpoint_durable"]) / "final"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {checkpoint_dir}")

    output_root = args.output_dir / args.run_id
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint contents
    shutil.copytree(checkpoint_dir, output_root / "checkpoint", dirs_exist_ok=True)

    # Persist manifest + metadata snapshot
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    metadata = {
        "run_id": args.run_id,
        "source_manifest": str(manifest_path),
        "checkpoint_dir": str(checkpoint_dir),
        "packaged_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Deployment package created at", output_root)


if __name__ == "__main__":
    main()
