"""Utilities for staging dataset shards onto local NVMe storage."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence
from urllib.parse import urlparse

from ...core.artifacts import ArtifactMeta, create_artifact
from ..catalog import CatalogError, ShardCatalog, ShardRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagedShard:
    """Mapping between source and staged shard locations."""

    source_uri: str
    staged_path: Path
    bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_uri": self.source_uri,
            "staged_path": str(self.staged_path),
            "bytes": self.bytes,
        }


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_local_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme in ("", "file")


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme:
        raise CatalogError(f"remote URIs are not supported by the NVMe staging stub: {uri}")
    return Path(uri)


def _copy_shard(record: ShardRecord, destination_dir: Path) -> StagedShard:
    source_path = _uri_to_path(record.uri)
    if not source_path.exists():
        raise CatalogError(f"shard not found: {record.uri}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    target_path = destination_dir / source_path.name
    # If the file already exists with the same size, reuse it.
    if target_path.exists() and (record.bytes is None or target_path.stat().st_size == record.bytes):
        LOGGER.debug("Reusing existing staged shard %s", target_path)
        size = target_path.stat().st_size
    else:
        LOGGER.debug("Copying shard %s -> %s", source_path, target_path)
        shutil.copy2(source_path, target_path)
        size = target_path.stat().st_size

    if record.bytes is not None and size != record.bytes:
        raise CatalogError(
            f"size mismatch for {record.uri}: manifest reports {record.bytes} bytes but staged file has {size} bytes"
        )

    return StagedShard(source_uri=record.uri, staged_path=target_path, bytes=size)


def stage_catalog(
    catalog: ShardCatalog,
    destination: Path,
) -> ArtifactMeta:
    """Materialise the shards listed in ``catalog`` to ``destination``.

    The function copies local shard files into a temporary artifact directory,
    writes a manifest describing the staged contents, and finalises the
    artifact using :func:`create_artifact`.
    """

    destination = Path(destination)

    def _producer(tmp_dir: Path) -> ArtifactMeta:
        staged_records: List[StagedShard] = []
        total_bytes = 0

        staging_dir = tmp_dir / "shards"
        for record in catalog:
            if not _is_local_uri(record.uri):
                raise CatalogError(
                    f"remote shard URIs are not yet supported by the staging stub: {record.uri}"
                )
            staged = _copy_shard(record, staging_dir)
            staged_records.append(staged)
            total_bytes += staged.bytes

        manifest = {
            "created_at": _now_utc(),
            "source_hash": catalog.content_hash(),
            "total_bytes": total_bytes,
            "shards": [item.to_dict() for item in staged_records],
        }
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

        payload_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        metadata = ArtifactMeta(
            kind="dataset",
            uri=str(destination.resolve()),
            hash=f"sha256:{payload_hash}",
            bytes=total_bytes,
            created_at=_now_utc(),
            inputs=catalog.uris(),
            labels={
                "source_catalog_hash": catalog.content_hash(),
                "staged_shards": str(len(staged_records)),
            },
        )

        return metadata

    return create_artifact(destination, _producer)


__all__ = [
    "StagedShard",
    "stage_catalog",
]
