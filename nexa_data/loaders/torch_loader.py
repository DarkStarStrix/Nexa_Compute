"""Wrappers over the core DataPipeline for external consumers."""

from __future__ import annotations

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.config import load_config  # type: ignore
from nexa_compute.data import DataPipeline  # type: ignore


def build_loader(config_path: Path, split: str):
    config = load_config(config_path)
    pipeline = DataPipeline(config.data)
    return pipeline.dataloader(split)
