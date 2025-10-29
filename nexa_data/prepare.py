"""Data preparation entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.config import load_config  # type: ignore
from nexa_compute.data import DataPipeline, DEFAULT_REGISTRY  # type: ignore


def prepare_from_config(config_path: Path, *, materialize_only: bool = False) -> Path:
    config = load_config(config_path)
    pipeline = DataPipeline(config.data, registry=DEFAULT_REGISTRY)
    run_dir = config.output_directory()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = pipeline.materialize_metadata(run_dir)
    if materialize_only:
        return metadata_path
    loaders = pipeline.available_dataloaders(["train", "validation", "test"])
    counts = {split: len(loader.dataset) for split, loader in loaders.items()}
    summary_path = run_dir / "dataset_summary.json"
    summary_path.write_text(str(counts), encoding="utf-8")
    return metadata_path


def build_dataloaders(config_path: Path, splits: Iterable[str]):
    config = load_config(config_path)
    pipeline = DataPipeline(config.data)
    return {split: pipeline.dataloader(split) for split in splits}
