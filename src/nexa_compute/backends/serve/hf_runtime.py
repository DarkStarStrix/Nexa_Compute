"""Hugging Face runtime serving backend (FastAPI fallback)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
from urllib.error import URLError
from urllib.request import urlopen

LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001

__all__ = ["HFRuntimeHandle", "start_server", "stop_server", "health_check"]


@dataclass
class HFRuntimeHandle:
    """Handle representing a running FastAPI inference server."""

    process: subprocess.Popen | None
    host: str
    port: int

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self, *, timeout: int = 15) -> None:
        stop_server(self, timeout=timeout)


def _build_command(checkpoint_path: Path, config_path: Optional[Path], host: str, port: int) -> list[str]:
    code = (
        "from nexa_inference.server import serve_model; "
        f"serve_model({repr(str(checkpoint_path))}, "
        f"{repr(str(config_path)) if config_path else 'None'}, "
        f"host={repr(host)}, port={port})"
    )
    return [sys.executable, "-c", code]


def start_server(
    checkpoint_path: Path,
    *,
    config_path: Optional[Path] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    env: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
) -> HFRuntimeHandle:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    config_path = Path(config_path) if config_path else None
    if config_path and not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    command = _build_command(checkpoint_path, config_path, host, port)
    LOGGER.info("[hf-runtime] launching server: %s", " ".join(command))

    if dry_run:
        return HFRuntimeHandle(process=None, host=host, port=port)

    env_vars = os.environ.copy()
    if env:
        env_vars.update({str(k): str(v) for k, v in env.items()})
    process = subprocess.Popen(command, env=env_vars)
    return HFRuntimeHandle(process=process, host=host, port=port)


def stop_server(handle: HFRuntimeHandle, *, timeout: int = 15) -> None:
    process = handle.process
    if process is None:
        LOGGER.info("[hf-runtime] dry-run handle; nothing to stop")
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        LOGGER.warning("[hf-runtime] graceful shutdown timed out; killing process")
        process.kill()


def health_check(handle: HFRuntimeHandle, *, timeout: float = 5.0) -> bool:
    if handle.process is None:
        LOGGER.info("[hf-runtime] health check skipped for dry-run handle")
        return False
    url = f"{handle.endpoint}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=timeout) as response:
                if response.status == 200:
                    data = json.load(response)
                    return data.get("status") == "healthy"
        except URLError:
            time.sleep(0.5)
    return False
