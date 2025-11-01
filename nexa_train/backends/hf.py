"""Hugging Face backend wrapper for the Nexa training launcher."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

from nexa_compute.training.hf_runner import HFTrainingConfig, run_training


_HF_FIELDS = {field.name for field in fields(HFTrainingConfig)}


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in params.items():
        if key not in _HF_FIELDS:
            print(f"[nexa-train][hf] Ignoring unknown parameter: {key}")
            continue
        if key == "tags" and isinstance(value, str):
            normalized[key] = [value]
        else:
            normalized[key] = value
    return normalized


def run(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Execute an HF training job using :mod:`nexa_compute.training.hf_runner`."""

    params = params or {}
    config_kwargs = _normalize_params(params)
    hf_config = HFTrainingConfig(**config_kwargs)
    result = run_training(hf_config)
    print(
        "[nexa-train][hf] run complete ::",
        result.get("run_id"),
        result.get("manifest_path"),
    )
    return result


__all__ = ["run"]


