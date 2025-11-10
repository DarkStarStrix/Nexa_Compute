"""Serving runner for launching inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from ..backends.serve.hf_runtime import (
    HFRuntimeHandle,
    health_check as hf_health_check,
    start_server as hf_start,
    stop_server as hf_stop,
)
from ..backends.serve.vllm import (
    VLLMServerHandle,
    health_check as vllm_health_check,
    start_server as vllm_start,
    stop_server as vllm_stop,
)

__all__ = ["ServeRunSpec", "ServeHandle", "ServeRunner"]


@dataclass(frozen=True)
class ServeRunSpec:
    """Configuration describing a serving deployment."""

    backend: str
    model_path: Path
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel: Optional[int] = None
    config_path: Optional[Path] = None
    env: Mapping[str, str] = None  # type: ignore[assignment]
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.env is None:  # pragma: no cover
            object.__setattr__(self, "env", {})


@dataclass(frozen=True)
class ServeHandle:
    """Handle returned by :class:`ServeRunner`."""

    backend: str
    handle: object


class ServeRunner:
    """Launch and manage serving processes."""

    def start(self, spec: ServeRunSpec) -> ServeHandle:
        backend = spec.backend.lower()
        if backend == "vllm":
            handle = vllm_start(
                spec.model_path,
                host=spec.host,
                port=spec.port,
                tensor_parallel=spec.tensor_parallel,
                env=spec.env,
                dry_run=spec.dry_run,
            )
            return ServeHandle(backend="vllm", handle=handle)
        if backend in {"hf", "hf_runtime"}:
            handle = hf_start(
                spec.model_path,
                config_path=spec.config_path,
                host=spec.host,
                port=spec.port,
                env=spec.env,
                dry_run=spec.dry_run,
            )
            return ServeHandle(backend="hf_runtime", handle=handle)
        raise ValueError(f"Unknown serve backend '{spec.backend}'")

    def stop(self, handle: ServeHandle) -> None:
        backend = handle.backend.lower()
        if backend == "vllm":
            vllm_stop(handle.handle)  # type: ignore[arg-type]
            return
        if backend == "hf_runtime":
            hf_stop(handle.handle)  # type: ignore[arg-type]
            return
        raise ValueError(f"Unknown serve backend '{handle.backend}'")

    def health_check(self, handle: ServeHandle) -> bool:
        backend = handle.backend.lower()
        if backend == "vllm":
            return vllm_health_check(handle.handle)  # type: ignore[arg-type]
        if backend == "hf_runtime":
            return hf_health_check(handle.handle)  # type: ignore[arg-type]
        raise ValueError(f"Unknown serve backend '{handle.backend}'")
