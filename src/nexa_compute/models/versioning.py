"""Model versioning and lineage tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    name: str
    version: str
    uri: str
    architecture: str
    hyperparameters: Dict[str, Any]
    dataset_versions: List[str]
    metrics: Dict[str, float]
    created_at: str
    created_by: str
    stage: ModelStage = ModelStage.DEVELOPMENT
    tags: List[str] = None # type: ignore

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "uri": self.uri,
            "architecture": self.architecture,
            "hyperparameters": self.hyperparameters,
            "dataset_versions": self.dataset_versions,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "stage": self.stage.value,
            "tags": self.tags,
        }

    @property
    def hash(self) -> str:
        """Compute a deterministic hash of the model metadata."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

