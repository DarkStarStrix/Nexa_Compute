"""Enhanced Model Registry with advanced metadata and lineage tracking."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexa_compute.core.registry import ModelRegistry as CoreRegistry, RegistryError, _connect, _utcnow
from .versioning import ModelStage, ModelVersion


class EnhancedModelRegistry(CoreRegistry):
    """Extended registry supporting rich metadata, lineage, and lifecycle stages."""

    def register_version(self, version: ModelVersion) -> None:
        """Register a fully specified model version."""
        # Store core metadata in existing tables
        core_meta = {
            "architecture": version.architecture,
            "hyperparameters": version.hyperparameters,
            "datasets": version.dataset_versions,
            "metrics": version.metrics,
            "stage": version.stage.value,
            "created_by": version.created_by,
        }
        
        self.register(
            name=version.name,
            uri=version.uri,
            meta=core_meta,
            version=version.version,
            tags=version.tags,
        )
        
        # Store additional lineage data (future extension: dedicated lineage table)
        self._record_lineage(version)

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Retrieve full metadata for a specific version."""
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT uri, meta_json, created_at FROM models WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
            
        if not row:
            return None
            
        meta = json.loads(row["meta_json"])
        return ModelVersion(
            name=name,
            version=version,
            uri=row["uri"],
            architecture=meta.get("architecture", "unknown"),
            hyperparameters=meta.get("hyperparameters", {}),
            dataset_versions=meta.get("datasets", []),
            metrics=meta.get("metrics", {}),
            created_at=row["created_at"],
            created_by=meta.get("created_by", "unknown"),
            stage=ModelStage(meta.get("stage", "development")),
            tags=[], # Tags are stored separately, would need another query
        )

    def transition_stage(self, name: str, version: str, stage: ModelStage) -> None:
        """Transition a model version to a new lifecycle stage."""
        current = self.get_version(name, version)
        if not current:
            raise RegistryError(f"Model {name} version {version} not found")
            
        # Update the JSON metadata
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM models WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
            meta = json.loads(row["meta_json"])
            meta["stage"] = stage.value
            
            conn.execute(
                "UPDATE models SET meta_json = ? WHERE name = ? AND version = ?",
                (json.dumps(meta), name, version),
            )

    def _record_lineage(self, version: ModelVersion) -> None:
        # Placeholder for graph-based lineage storage
        pass

    def compare_versions(self, name: str, v1: str, v2: str) -> Dict[str, Any]:
        """Compare two versions of a model."""
        m1 = self.get_version(name, v1)
        m2 = self.get_version(name, v2)
        
        if not m1 or not m2:
            raise RegistryError("One or both versions not found")
            
        return {
            "architecture_match": m1.architecture == m2.architecture,
            "hyperparameters_diff": self._diff_dicts(m1.hyperparameters, m2.hyperparameters),
            "metrics_diff": self._diff_dicts(m1.metrics, m2.metrics),
            "dataset_changes": {
                "added": list(set(m2.dataset_versions) - set(m1.dataset_versions)),
                "removed": list(set(m1.dataset_versions) - set(m2.dataset_versions)),
            },
        }

    @staticmethod
    def _diff_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        diff = {}
        all_keys = set(d1.keys()) | set(d2.keys())
        for k in all_keys:
            if k not in d1:
                diff[k] = {"old": None, "new": d2[k]}
            elif k not in d2:
                diff[k] = {"old": d1[k], "new": None}
            elif d1[k] != d2[k]:
                diff[k] = {"old": d1[k], "new": d2[k]}
        return diff

