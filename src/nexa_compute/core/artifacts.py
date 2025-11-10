"""Utilities for writing and promoting pipeline artifacts atomically.

This module implements the artifact protocol described in ``docs/Spec_V2.md``.
Artifacts are materialised into a temporary directory, synchronised to stable
storage, atomically renamed into place, and finally marked COMPLETE. Pointer
files (such as ``latest.txt``) are only updated once an artifact exists with a
COMPLETE marker.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

COMPLETE_MARKER = "COMPLETE"
META_FILENAME = "meta.json"
TEMP_SUFFIX = ".tmp"


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be created or validated."""


@dataclass(frozen=True)
class ArtifactMeta:
    """Metadata captured for every materialised artifact.

    Parameters
    ----------
    kind:
        High-level type of artifact (checkpoint, eval_report, dataset, index, model_card).
    uri:
        Canonical URI where the artifact is stored (local path, S3 URI, etc.).
    hash:
        Hash of the artifact contents (``sha256:<digest>``).
    bytes:
        Size of the artifact payload in bytes.
    created_at:
        ISO8601 timestamp.
    inputs:
        URIs for the input datasets or artifacts that produced this artifact.
    labels:
        Additional key-value metadata for downstream consumers.
    """

    kind: str
    uri: str
    hash: str
    bytes: int
    created_at: str
    inputs: Sequence[str] = field(default_factory=list)
    labels: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return metadata as a serialisable dictionary."""

        return {
            "kind": self.kind,
            "uri": self.uri,
            "hash": self.hash,
            "bytes": int(self.bytes),
            "created_at": self.created_at,
            "inputs": list(self.inputs),
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ArtifactMeta":
        """Instantiate metadata from a parsed JSON dictionary."""

        try:
            return cls(
                kind=str(payload["kind"]),
                uri=str(payload["uri"]),
                hash=str(payload["hash"]),
                bytes=int(payload["bytes"]),
                created_at=str(payload["created_at"]),
                inputs=tuple(str(item) for item in payload.get("inputs", []) or []),
                labels={str(k): str(v) for k, v in (payload.get("labels") or {}).items()},
            )
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ArtifactError(f"missing required metadata field: {exc}") from exc


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tmp_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.name}{TEMP_SUFFIX}")


def _ensure_removed(path: Path) -> None:
    if path.exists():
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


def _fsync_path(path: Path) -> None:
    if path.is_dir():
        for child in path.iterdir():
            _fsync_path(child)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:  # pragma: no cover - ensure descriptor closed if fsync fails
            os.close(fd)
    else:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


def _write_meta(path: Path, meta: ArtifactMeta) -> None:
    payload = meta.to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _fsync_path(path)


def _finalise(tmp_dir: Path, final_dir: Path) -> None:
    _fsync_path(tmp_dir)
    if final_dir.exists():
        raise ArtifactError(f"artifact already exists: {final_dir}")
    tmp_dir.rename(final_dir)
    complete_marker = final_dir / COMPLETE_MARKER
    complete_marker.touch()
    _fsync_path(complete_marker)
    _fsync_path(final_dir)


def create_artifact(
    destination: Path,
    producer: Callable[[Path], ArtifactMeta | tuple[ArtifactMeta, MutableMapping[str, object]]],
    *,
    metadata: ArtifactMeta | None = None,
) -> ArtifactMeta:
    """Create an artifact by materialising it into a temporary directory.

    Parameters
    ----------
    destination:
        Final directory where the artifact should reside.
    producer:
        Callable that receives the temporary directory and writes all payload
        files. It must either return an :class:`ArtifactMeta` instance or a
        tuple ``(meta, extra_paths)`` where ``extra_paths`` is a mapping of
        relative destination path -> ``Path`` objects that should be hard-linked
        or copied into the artifact before finalisation.
    metadata:
        Optional pre-computed metadata. If provided, the producer may return
        ``None`` to signal that this metadata should be used verbatim.

    Returns
    -------
    ArtifactMeta
        Metadata associated with the written artifact.
    """

    final_dir = destination.resolve()
    tmp_dir = _tmp_path(final_dir)
    _ensure_removed(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    produced_meta: ArtifactMeta | None = metadata
    try:
        result = producer(tmp_dir)
        extra_assets: MutableMapping[str, object] | None = None
        if isinstance(result, tuple):
            produced_meta, extra_assets = result
            if extra_assets:
                for rel_path, source in extra_assets.items():
                    target = tmp_dir / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(source, Path):
                        shutil.copy2(source, target)
                    else:
                        target.write_bytes(source)  # pragma: no cover - rarely used
        elif isinstance(result, ArtifactMeta):
            produced_meta = result
        elif result is None and metadata is not None:
            produced_meta = metadata
        else:
            raise ArtifactError(
                "producer must return ArtifactMeta, (ArtifactMeta, extra_assets) or None when metadata is provided"
            )
    except Exception:
        _ensure_removed(tmp_dir)
        raise

    if produced_meta is None:
        raise ArtifactError("artifact metadata was not provided by producer and no metadata argument supplied")

    _write_meta(tmp_dir / META_FILENAME, produced_meta)
    _finalise(tmp_dir, final_dir)
    return produced_meta


def load_artifact_meta(artifact_path: Path) -> ArtifactMeta:
    """Load artifact metadata from ``meta.json``."""

    meta_path = artifact_path / META_FILENAME
    if not meta_path.exists():
        raise ArtifactError(f"missing metadata file: {meta_path}")
    payload = json.loads(meta_path.read_text())
    return ArtifactMeta.from_dict(payload)


def is_complete(artifact_path: Path) -> bool:
    """Return ``True`` if an artifact exists with a COMPLETE marker."""

    marker = artifact_path / COMPLETE_MARKER
    return marker.exists()


def promote(pointer: Path, artifact_path: Path) -> None:
    """Promote an artifact by updating the pointer file.

    The pointer file is only updated if the artifact path contains a COMPLETE
    marker. The pointer file is written atomically via a temporary file.
    """

    artifact_dir = artifact_path.resolve()
    if not is_complete(artifact_dir):
        raise ArtifactError(f"cannot promote incomplete artifact: {artifact_dir}")

    pointer = pointer.resolve()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp_pointer = pointer.with_suffix(pointer.suffix + TEMP_SUFFIX)
    tmp_pointer.write_text(str(artifact_dir))
    _fsync_path(tmp_pointer)
    tmp_pointer.replace(pointer)
    _fsync_path(pointer)


__all__ = [
    "ArtifactMeta",
    "ArtifactError",
    "COMPLETE_MARKER",
    "META_FILENAME",
    "create_artifact",
    "is_complete",
    "load_artifact_meta",
    "promote",
]

