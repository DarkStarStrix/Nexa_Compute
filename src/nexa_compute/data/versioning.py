"""Data versioning using Content-Addressable Storage (CAS)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexa_compute.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DatasetVersion:
    """Immutable record of a dataset version."""
    name: str
    version: str # sha256 hash of contents
    files: Dict[str, str] # path -> sha256
    metadata: Dict[str, Any]
    created_at: str
    created_by: str
    parent_version: Optional[str] = None


class DataVersionControl:
    """Manages dataset versions and file storage."""

    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root.resolve()
        self.blob_store = self.storage_root / "blobs"
        self.meta_store = self.storage_root / "meta"
        self.blob_store.mkdir(parents=True, exist_ok=True)
        self.meta_store.mkdir(parents=True, exist_ok=True)

    def commit(
        self,
        name: str,
        source_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional[str] = None,
    ) -> DatasetVersion:
        """Create a new version from a directory."""
        source_dir = source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        file_map = {}
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(source_dir)
                file_hash = self._store_blob(file_path)
                file_map[str(rel_path)] = file_hash

        # Compute version hash
        version_payload = {
            "name": name,
            "files": file_map,
            "metadata": metadata or {},
            "parent": parent,
        }
        version_hash = hashlib.sha256(json.dumps(version_payload, sort_keys=True).encode()).hexdigest()

        from datetime import datetime, timezone
        version = DatasetVersion(
            name=name,
            version=version_hash,
            files=file_map,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=os.getenv("USER", "unknown"),
            parent_version=parent,
        )
        
        self._save_version_meta(version)
        LOGGER.info("dataset_committed", extra={"name": name, "version": version_hash})
        return version

    def checkout(self, name: str, version_hash: str, target_dir: Path) -> None:
        """Restore a dataset version to a directory."""
        version = self._load_version_meta(name, version_hash)
        target_dir = target_dir.resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for rel_path, blob_hash in version.files.items():
            blob_path = self._get_blob_path(blob_hash)
            dest_path = target_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(blob_path, dest_path)
            
        LOGGER.info("dataset_checkout", extra={"name": name, "version": version_hash, "target": str(target_dir)})

    def _store_blob(self, file_path: Path) -> str:
        """Store a file in the blob store (CAS)."""
        # Calculate hash first
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        
        # Store if not exists
        blob_path = self._get_blob_path(digest)
        if not blob_path.exists():
            # Ensure parent directory exists (sharding)
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use a temp file and atomic rename for safety
            tmp_path = blob_path.with_suffix(".tmp")
            shutil.copy2(file_path, tmp_path)
            tmp_path.rename(blob_path)
            
        return digest

    def _get_blob_path(self, digest: str) -> Path:
        # Use 2-level sharding to avoid too many files in one dir
        return self.blob_store / digest[:2] / digest[2:]

    def _save_version_meta(self, version: DatasetVersion) -> None:
        meta_path = self.meta_store / version.name / f"{version.version}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "name": version.name,
            "version": version.version,
            "files": version.files,
            "metadata": version.metadata,
            "created_at": version.created_at,
            "created_by": version.created_by,
            "parent_version": version.parent_version,
        }
        
        with meta_path.open("w") as f:
            json.dump(payload, f, indent=2)

    def _load_version_meta(self, name: str, version_hash: str) -> DatasetVersion:
        meta_path = self.meta_store / name / f"{version_hash}.json"
        if not meta_path.exists():
            raise ValueError(f"Version {version_hash} for dataset {name} not found")
            
        with meta_path.open("r") as f:
            data = json.load(f)
            
        return DatasetVersion(
            name=data["name"],
            version=data["version"],
            files=data["files"],
            metadata=data["metadata"],
            created_at=data["created_at"],
            created_by=data["created_by"],
            parent_version=data.get("parent_version"),
        )

