"""Training runner that delegates to registered backends and schedulers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import yaml

from ..backends.schedule.local import LocalScheduler
from ..backends.train.axolotl import AxolotlBackend
from ..backends.train.hf import run as hf_run
from ..core.artifacts import ArtifactMeta

LOGGER = logging.getLogger(__name__)

__all__ = ["TrainRunSpec", "TrainResult", "TrainRunner"]


@dataclass(frozen=True)
class TrainRunSpec:
    """Configuration describing a training run."""

    backend: str
    scheduler: str
    params: Mapping[str, object]
    artifact_path: Path
    env: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.env is None:  # pragma: no cover - dataclass quirk
            object.__setattr__(self, "env", {})


@dataclass(frozen=True)
class TrainResult:
    """Wrapper for training results."""

    artifact: ArtifactMeta


class TrainRunner:
    """Execute training runs using the configured backend and scheduler."""

    def __init__(self) -> None:
        self._local_scheduler = LocalScheduler()

    def run(self, spec: TrainRunSpec) -> TrainResult:
        backend = spec.backend.lower()
        scheduler = spec.scheduler.lower()

        if scheduler == "local":
            artifact = self._run_local(backend, spec)
            return TrainResult(artifact=artifact)
        if scheduler in {"slurm", "k8s"}:
            raise NotImplementedError(f"Scheduler '{scheduler}' is not implemented yet")
        raise ValueError(f"Unknown scheduler '{spec.scheduler}'")

    def _run_local(self, backend: str, spec: TrainRunSpec) -> ArtifactMeta:
        if backend == "axolotl":
            recipe_path = spec.params.get("recipe_path")
            recipe = _load_recipe(recipe_path)
            overrides = spec.params.get("config", {})
            dry_run = bool(spec.params.get("dry_run", False))
            ax_backend = AxolotlBackend(recipe=recipe)
            return ax_backend.run(overrides, artifact_dir=spec.artifact_path, env=spec.env, dry_run=dry_run)

        if backend == "hf":
            return hf_run(spec.params, artifact_dir=spec.artifact_path)

        raise ValueError(f"Unknown training backend '{backend}'")


def _load_recipe(recipe_path: Optional[object]) -> Optional[Mapping[str, object]]:
    if not recipe_path:
        return None
    path = Path(str(recipe_path))
    if not path.exists():
        raise FileNotFoundError(f"Axolotl recipe not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
