#!/usr/bin/env python3
"""Promote a packaged checkpoint to the deployment location."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from scripts.python import project_root

PROJECT_ROOT = project_root(Path(__file__))
SRC = PROJECT_ROOT / "src"


def setup_paths() -> None:
    """Ensure project modules can be imported."""

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


setup_paths()

DEPLOY_ROOT = Path("/mnt/nexa_durable/deploy")


def parse_args() -> argparse.Namespace:
    """Return CLI parameters for promotion of a packaged model."""

    parser = argparse.ArgumentParser(description="Deploy a trained NexaCompute model")
    parser.add_argument("run_id", type=str, help="Run identifier to promote")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/deployment"), help="Packaged artifacts directory")
    parser.add_argument("--deploy-root", type=Path, default=DEPLOY_ROOT, help="Deployment target directory")
    return parser.parse_args()


def load_manifest(package_dir: Path) -> dict:
    """Return deployment metadata if present."""

    manifest_path = package_dir / "manifest.json"
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def deploy_checkpoint(source_dir: Path, target_root: Path, manifest: dict) -> None:
    """Copy packaged checkpoint to the deployment root and persist manifest."""

    target_root.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target_root / "current_model"
    if target_checkpoint.exists():
        shutil.rmtree(target_checkpoint)
    shutil.copytree(source_dir, target_checkpoint)
    (target_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✅ Deployed checkpoint to {target_checkpoint}")


def main() -> None:
    """Promote a packaged run to the deployment target."""

    args = parse_args()
    package_dir = args.artifact_root / args.run_id
    if not package_dir.exists():
        raise FileNotFoundError(f"Packaged run not found: {package_dir}")

    checkpoint_dir = package_dir / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {checkpoint_dir}")

    manifest = load_manifest(package_dir)
    deploy_checkpoint(checkpoint_dir, args.deploy_root, manifest)


if __name__ == "__main__":
    main()
