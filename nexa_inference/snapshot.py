"""Local snapshot utilities for offline search."""

from __future__ import annotations

import shutil
from pathlib import Path


def create_snapshot(source_dir: Path, snapshot_dir: Path) -> Path:
    """Copy vector DB artifacts to snapshot directory."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.glob("**/*"):
        target = snapshot_dir / path.relative_to(source_dir)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return snapshot_dir

