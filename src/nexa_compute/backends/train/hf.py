"""Hugging Face training backend wrapper with artifact emission."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from nexa_train.backends import hf as hf_backend

from ...core.artifacts import ArtifactMeta, create_artifact

LOGGER = logging.getLogger(__name__)

__all__ = ["run"]


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(params: Mapping[str, object], *, artifact_dir: Path) -> ArtifactMeta:
    """Execute the HF backend and wrap results in an artifact."""

    params_dict = dict(params)
    LOGGER.info("[hf-backend] starting run with params %s", params_dict)
    result = hf_backend.run(params_dict)
    manifest = result.get("manifest", {})
    run_id = result.get("run_id")
    checkpoint_path_str = manifest.get("checkpoint_durable")
    if not checkpoint_path_str:
        raise RuntimeError("HF backend did not return 'checkpoint_durable' in manifest")
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint path not found: {checkpoint_path}")

    manifest_path = result.get("manifest_path")

    def _producer(tmp_dir: Path) -> ArtifactMeta:
        target_checkpoint = tmp_dir / "checkpoint"
        if checkpoint_path.is_dir():
            shutil.copytree(checkpoint_path, target_checkpoint, dirs_exist_ok=True)
        else:
            target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, target_checkpoint)

        manifest_file = tmp_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if manifest_path:
            original_manifest = Path(manifest_path)
            if original_manifest.exists():
                shutil.copy2(original_manifest, tmp_dir / original_manifest.name)

        payload_hash = hashlib.sha256()
        total_bytes = 0
        for file_path in tmp_dir.rglob("*"):
            if file_path.is_file():
                data = file_path.read_bytes()
                payload_hash.update(data)
                total_bytes += len(data)

        labels = {"backend": "hf"}
        if run_id:
            labels["run_id"] = str(run_id)

        metadata = ArtifactMeta(
            kind="checkpoint",
            uri=str(Path(artifact_dir).resolve()),
            hash=f"sha256:{payload_hash.hexdigest()}",
            bytes=total_bytes,
            created_at=_now_utc(),
            inputs=[str(manifest.get("dataset"))] if manifest.get("dataset") else [],
            labels=labels,
        )
        return metadata

    return create_artifact(Path(artifact_dir), _producer)
