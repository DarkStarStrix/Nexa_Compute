"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from ..config.schema import CheckpointConfig


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: str | Path, *, filename: str = "model.pt") -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | None = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def checkpoint_path(config: CheckpointConfig, *, epoch: int | None = None) -> Path:
    postfix = f"epoch{epoch}" if epoch is not None else "latest"
    return Path(config.dir) / f"checkpoint_{postfix}.pt"

