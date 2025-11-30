"""Shared utilities for CLI scripts."""

from __future__ import annotations

from pathlib import Path


def project_root(current_file: Path) -> Path:
    """Locate repository root by walking up until pyproject.toml is found."""
    path = current_file.resolve()
    for ancestor in [path, *path.parents]:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    # Fallback: two levels up from file
    return path.parents[2]

