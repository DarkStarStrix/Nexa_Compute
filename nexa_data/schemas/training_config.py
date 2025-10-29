"""Re-export of the core TrainingConfig schema for data pipelines."""

from __future__ import annotations

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.config.schema import TrainingConfig  # type: ignore

__all__ = ["TrainingConfig"]
