"""Utilities for locating NexaCompute project directories."""

from __future__ import annotations

from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Return the project root by climbing directories from ``start``."""

    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for parent in [candidate, *candidate.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # Fallback to the topmost directory we inspected.
    return candidate


def scripts_root(start: Path | None = None) -> Path:
    """Return the ``scripts`` directory root relative to ``start``."""

    root = project_root(start=start)
    return root / "scripts"


