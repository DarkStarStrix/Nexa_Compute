"""Training entrypoints wrapping the core training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.orchestration import TrainingPipeline  # type: ignore
from nexa_compute.config import load_config  # type: ignore

from .utils import save_run_manifest


def run_training_job(config_path: Path, overrides: Optional[List[str]] = None) -> None:
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
