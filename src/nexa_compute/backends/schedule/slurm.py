"""Slurm scheduler wrapper for NexaCompute pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nexa_infra.slurm import SlurmBatchArtifacts, prepare_slurm_batch

__all__ = ["SlurmScheduler", "SlurmBatchArtifacts"]


@dataclass
class SlurmScheduler:
    """Minimal wrapper delegating to :mod:`nexa_infra.slurm`."""

    config_path: Path

    def prepare(self, *, submit: bool = False, output_dir: Optional[Path] = None) -> SlurmBatchArtifacts:
        """Render Slurm batch assets and optionally submit the job."""

        return prepare_slurm_batch(self.config_path, submit=submit, output_dir=output_dir)
