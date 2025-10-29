"""Shared utilities for training workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.config import load_config  # type: ignore


def load_training_config(config_path: Path, overrides: Optional[List[str]] = None):
    return load_config(config_path, overrides=overrides or [])


def save_run_manifest(run_dir: Path, manifest: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path
