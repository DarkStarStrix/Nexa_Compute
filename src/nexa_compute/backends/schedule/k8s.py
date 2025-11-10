"""Kubernetes scheduler stub for NexaCompute."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = ["KubernetesScheduler", "KubernetesJob"]


@dataclass(frozen=True)
class KubernetesJob:
    """Placeholder for Kubernetes job specifications."""

    manifest_path: Path
    namespace: str = "default"


class KubernetesScheduler:
    """Stub scheduler that raises ``NotImplementedError``."""

    def submit(self, job: KubernetesJob, *, overrides: Mapping[str, object] | None = None) -> None:  # pragma: no cover - stub
        raise NotImplementedError("Kubernetes scheduler integration is not implemented yet")

    def status(self, job_name: str) -> str:  # pragma: no cover - stub
        raise NotImplementedError("Kubernetes scheduler integration is not implemented yet")

    def cancel(self, job_name: str) -> None:  # pragma: no cover - stub
        raise NotImplementedError("Kubernetes scheduler integration is not implemented yet")
