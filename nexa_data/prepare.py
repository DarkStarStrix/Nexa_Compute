"""Data preparation entrypoints."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.config import load_config  # type: ignore
from nexa_compute.core.artifacts import ArtifactMeta, create_artifact  # type: ignore
from nexa_compute.data import DataPipeline, DEFAULT_REGISTRY  # type: ignore
from nexa_compute.data.catalog import ShardCatalog  # type: ignore
from nexa_compute.data.staging.nvme_cache import stage_catalog  # type: ignore


_DEF_METADATA_NAME = "dataset_metadata.json"
_DEF_SUMMARY_NAME = "dataset_summary.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def prepare_from_config(config_path: Path, *, materialize_only: bool = False) -> ArtifactMeta:
    config = load_config(config_path)
    pipeline = DataPipeline(config.data, registry=DEFAULT_REGISTRY)
    run_dir = config.output_directory()
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = pipeline.materialize_metadata(run_dir)
    metadata_target = run_dir / _DEF_METADATA_NAME
    shutil.copy2(metadata_path, metadata_target)

    summary_path = run_dir / _DEF_SUMMARY_NAME
    counts = {}
    if not materialize_only:
        loaders = pipeline.available_dataloaders(["train", "validation", "test"])
        counts = {split: len(loader.dataset) for split, loader in loaders.items()}
    summary_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    artifact_dir = run_dir / "artifacts" / "dataset"

    def _producer(tmp_dir: Path) -> ArtifactMeta:
        tmp_metadata = tmp_dir / metadata_target.name
        tmp_summary = tmp_dir / summary_path.name
        shutil.copy2(metadata_target, tmp_metadata)
        shutil.copy2(summary_path, tmp_summary)

        hasher = hashlib.sha256()
        metadata_bytes = tmp_metadata.read_bytes()
        summary_bytes = tmp_summary.read_bytes()
        hasher.update(metadata_bytes)
        hasher.update(summary_bytes)

        return ArtifactMeta(
            kind="dataset",
            uri=str(artifact_dir.resolve()),
            hash=f"sha256:{hasher.hexdigest()}",
            bytes=len(metadata_bytes) + len(summary_bytes),
            created_at=_now_utc(),
            inputs=[str(config_path)],
            labels={
                "source": "nexa_data.prepare",
                "materialize_only": str(materialize_only).lower(),
            },
        )

    return create_artifact(artifact_dir, _producer)


def stage_catalog_to_nvme(catalog_path: Path, destination: Path) -> ArtifactMeta:
    catalog = ShardCatalog.from_jsonl(catalog_path)
    catalog.validate()
    metadata = stage_catalog(catalog, destination)
    return metadata


def build_dataloaders(config_path: Path, splits: Iterable[str]):
    config = load_config(config_path)
    pipeline = DataPipeline(config.data)
    return {split: pipeline.dataloader(split) for split in splits}
