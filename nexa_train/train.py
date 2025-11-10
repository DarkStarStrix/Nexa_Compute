"""Training entrypoints wrapping the core training pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.orchestration import TrainingPipeline  # type: ignore
from nexa_compute.config import load_config  # type: ignore
from nexa_compute.core.artifacts import ArtifactMeta, create_artifact  # type: ignore

from .utils import save_run_manifest


_DEF_CHECKPOINT_ARTIFACT_DIR = "artifacts/checkpoints"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_training_job(config_path: Path, overrides: Optional[List[str]] = None) -> ArtifactMeta:
    config = load_config(config_path, overrides=overrides or [])
    pipeline = TrainingPipeline(config)
    artifacts = pipeline.run()

    manifest = {
        "config": str(config_path),
        "run_dir": str(artifacts.run_dir),
        "metrics": artifacts.metrics,
        "checkpoint": str(artifacts.checkpoint) if artifacts.checkpoint else None,
    }
    save_run_manifest(config.output_directory(), manifest)
    print(f"[nexa-train] run complete at {artifacts.run_dir}")

    artifact_dir = config.output_directory() / _DEF_CHECKPOINT_ARTIFACT_DIR

    def _producer(tmp_dir: Path) -> ArtifactMeta:
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        hasher = hashlib.sha256()
        total_bytes = 0

        if artifacts.checkpoint and Path(artifacts.checkpoint).exists():
            checkpoint_path = Path(artifacts.checkpoint)
            target_checkpoint = tmp_dir / checkpoint_path.name
            if checkpoint_path.is_dir():
                shutil.copytree(checkpoint_path, target_checkpoint, dirs_exist_ok=True)
                for file_path in target_checkpoint.rglob("*"):
                    if file_path.is_file():
                        data = file_path.read_bytes()
                        hasher.update(data)
                        total_bytes += len(data)
            else:
                shutil.copy2(checkpoint_path, target_checkpoint)
                data = target_checkpoint.read_bytes()
                hasher.update(data)
                total_bytes += len(data)
        else:
            placeholder = tmp_dir / "checkpoint.txt"
            placeholder.write_text("No checkpoint emitted by pipeline\n", encoding="utf-8")
            data = placeholder.read_bytes()
            hasher.update(data)
            total_bytes += len(data)

        manifest_bytes = manifest_path.read_bytes()
        hasher.update(manifest_bytes)
        total_bytes += len(manifest_bytes)

        return ArtifactMeta(
            kind="checkpoint",
            uri=str(artifact_dir.resolve()),
            hash=f"sha256:{hasher.hexdigest()}",
            bytes=total_bytes,
            created_at=_now_utc(),
            inputs=[str(config_path)],
            labels={"source": "nexa_train.train"},
        )

    return create_artifact(artifact_dir, _producer)
