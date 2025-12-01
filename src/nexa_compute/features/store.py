"""Feature Store abstraction for managing ML features."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

LOGGER = logging.getLogger(__name__)


class FeatureType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    VECTOR = "vector"


@dataclass
class FeatureDefinition:
    name: str
    type: FeatureType
    description: str
    tags: Dict[str, str]
    owner: str


class FeatureStore:
    """Manages feature definitions and retrieval."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path.resolve()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.definitions: Dict[str, FeatureDefinition] = {}
        self._load_definitions()

    def register_feature(
        self,
        name: str,
        type: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        owner: str = "system",
    ) -> None:
        """Register a new feature definition."""
        try:
            ft = FeatureType(type)
        except ValueError:
            raise ValueError(f"Invalid feature type: {type}. Valid: {[t.value for t in FeatureType]}")

        defn = FeatureDefinition(
            name=name,
            type=ft,
            description=description,
            tags=tags or {},
            owner=owner,
        )
        self.definitions[name] = defn
        self._save_definition(defn)
        LOGGER.info("feature_registered", extra={"name": name, "type": type})

    def get_feature(self, name: str) -> FeatureDefinition:
        if name not in self.definitions:
            raise KeyError(f"Feature '{name}' not found")
        return self.definitions[name]

    def put_features(self, entity_id: str, features: Dict[str, Any]) -> None:
        """Store feature values for an entity (Online Store)."""
        # Simplified implementation: write to JSON file per entity
        # Production: write to Redis/DynamoDB
        entity_path = self.storage_path / "values" / f"{entity_id}.json"
        entity_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate
        for name, value in features.items():
            if name not in self.definitions:
                raise KeyError(f"Feature '{name}' not registered")
            # Type check could go here
            
        with entity_path.open("w") as f:
            json.dump(features, f)

    def get_features(self, entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
        """Retrieve feature values for an entity."""
        entity_path = self.storage_path / "values" / f"{entity_id}.json"
        if not entity_path.exists():
            return {name: None for name in feature_names}
            
        with entity_path.open("r") as f:
            data = json.load(f)
            
        return {name: data.get(name) for name in feature_names}

    def _save_definition(self, defn: FeatureDefinition) -> None:
        path = self.storage_path / "definitions" / f"{defn.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(
                {
                    "name": defn.name,
                    "type": defn.type.value,
                    "description": defn.description,
                    "tags": defn.tags,
                    "owner": defn.owner,
                },
                f,
                indent=2,
            )

    def _load_definitions(self) -> None:
        def_dir = self.storage_path / "definitions"
        if not def_dir.exists():
            return
            
        for path in def_dir.glob("*.json"):
            with path.open("r") as f:
                data = json.load(f)
                self.definitions[data["name"]] = FeatureDefinition(
                    name=data["name"],
                    type=FeatureType(data["type"]),
                    description=data["description"],
                    tags=data["tags"],
                    owner=data["owner"],
                )

# Global instance
import os
_FEATURE_STORE = None

def get_feature_store() -> FeatureStore:
    global _FEATURE_STORE
    if _FEATURE_STORE is None:
        path = Path(os.getenv("FEATURE_STORE_ROOT", "data/feature_store"))
        _FEATURE_STORE = FeatureStore(path)
    return _FEATURE_STORE

