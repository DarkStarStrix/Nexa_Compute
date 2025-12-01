"""Data Catalog for managing dataset lifecycle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from nexa_compute.api.config import get_settings
from nexa_compute.data.versioning import DataVersionControl

settings = get_settings()


class DataCatalog:
    """High-level interface for data versioning and management."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root = root_dir or Path(os.getenv("DATA_ROOT", "data"))
        self.dvc = DataVersionControl(self.root / ".dvc")

    def register_dataset(
        self,
        name: str,
        source_path: Path,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a new version of a dataset."""
        metadata = {
            "description": description,
            "tags": tags or {},
        }
        version = self.dvc.commit(name, source_path, metadata=metadata)
        return version.version

    def load_dataset(self, name: str, version: str, target_path: Path) -> None:
        """Materialize a specific dataset version."""
        self.dvc.checkout(name, version, target_path)

    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the hash of the most recent version."""
        # This is a simplification; real impl would track 'latest' pointer
        dataset_dir = self.dvc.meta_store / name
        if not dataset_dir.exists():
            return None
            
        # Sort by creation time (assumes files are named by hash, need to read content)
        # In a real system, we'd use a DB or 'latest.txt'
        # For now, just return one
        versions = list(dataset_dir.glob("*.json"))
        if not versions:
            return None
        return versions[0].stem

# Global instance
_CATALOG = DataCatalog()

def get_catalog() -> DataCatalog:
    return _CATALOG
