#!/usr/bin/env python3
"""Promote a packaged checkpoint to the deployment location."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from nexa_compute.utils.storage import get_storage  # type: ignore

DEPLOY_ROOT = Path("/mnt/nexa_durable/deploy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a trained NexaCompute model")
    parser.add_argument("run_id", type=str, help="Run identifier to promote")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/deployment"), help="Packaged artifacts directory")
    parser.add_argument("--deploy-root", type=Path, default=DEPLOY_ROOT, help="Deployment target directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.artifact_root / args.run_id
    if not package_dir.exists():
        raise FileNotFoundError(f"Packaged run not found: {package_dir}")

    checkpoint_dir = package_dir / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {checkpoint_dir}")

    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    target_root = args.deploy_root
    target_root.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target_root / "current_model"
    if target_checkpoint.exists():
        shutil.rmtree(target_checkpoint)
    shutil.copytree(checkpoint_dir, target_checkpoint)

    (target_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✅ Deployed {args.run_id} to {target_checkpoint}")


if __name__ == "__main__":
    main()
