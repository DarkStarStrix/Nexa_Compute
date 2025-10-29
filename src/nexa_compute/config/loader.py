"""Helpers for reading and writing configuration files."""

from __future__ import annotations

import datetime as dt
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

from .schema import TrainingConfig


def _merge_dict(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _merge_dict(deepcopy(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _parse_override(override: str) -> Dict[str, Any]:
    if "=" not in override:
        raise ValueError(f"Override '{override}' must be in key=value format")
    key, raw_value = override.split("=", 1)
    # Try to interpret JSON so we can support numbers, lists, bools
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    nested_keys = key.split(".")
    current: Dict[str, Any] = {}
    cursor = current
    for nested_key in nested_keys[:-1]:
        cursor[nested_key] = {}
        cursor = cursor[nested_key]
    cursor[nested_keys[-1]] = value
    return current


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> TrainingConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload = deepcopy(payload)

    if overrides:
        for override in overrides:
            payload = _merge_dict(payload, _parse_override(override))

    config = TrainingConfig.model_validate(payload)
    return config


def save_run_config(config: TrainingConfig, output_dir: str | Path, *, filename: str = "config_resolved.yaml") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    serializable = json.loads(config.model_dump_json())
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)
    # Persist metadata snapshot
    metadata = {
        "saved_at": dt.datetime.utcnow().isoformat() + "Z",
        "output_dir": str(output_dir),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path
