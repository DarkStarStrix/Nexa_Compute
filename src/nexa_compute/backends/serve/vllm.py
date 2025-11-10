"""vLLM serving backend for NexaCompute."""

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
DEFAULT_PORT = 8000
DEFAULT_MODULE = "vllm.entrypoints.openai.api_server"

__all__ = ["ServeError", "VLLMServerHandle", "start_server", "stop_server", "health_check"]


class ServeError(RuntimeError):
    """Raised when a serving operation fails."""


@dataclass
class VLLMServerHandle:
    """Handle representing a running vLLM server."""

    process: subprocess.Popen | None
    host: str
    port: int

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self, *, timeout: int = 30) -> None:
        stop_server(self, timeout=timeout)


def _build_command(model_path: Path, host: str, port: int, tensor_parallel: Optional[int]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        DEFAULT_MODULE,
        "--model",
        str(model_path),
        "--port",
        str(port),
        "--host",
        host,
    ]
    if tensor_parallel and tensor_parallel > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel)])
    return cmd


def start_server(
    checkpoint_path: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tensor_parallel: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
) -> VLLMServerHandle:
    """Launch a vLLM server for the supplied checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ServeError(f"checkpoint not found: {checkpoint_path}")

    command = _build_command(checkpoint_path, host, port, tensor_parallel)
    LOGGER.info("[vllm] launching server: %s", " ".join(command))

    if dry_run:
        return VLLMServerHandle(process=None, host=host, port=port)

    env_vars = os.environ.copy()
    if env:
        env_vars.update({str(key): str(value) for key, value in env.items()})
    process = subprocess.Popen(command, env=env_vars)
    return VLLMServerHandle(process=process, host=host, port=port)


def stop_server(handle: VLLMServerHandle, *, timeout: int = 30) -> None:
    """Terminate a running vLLM server."""

    process = handle.process
    if process is None:
        LOGGER.info("[vllm] dry-run handle; nothing to stop")
        return

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        LOGGER.warning("[vllm] graceful shutdown timed out; killing process")
        process.kill()


def health_check(handle: VLLMServerHandle, *, timeout: float = 5.0) -> bool:
    """Return ``True`` if the server responds to ``/v1/models``."""

    url = f"{handle.endpoint}/v1/models"
    if handle.process is None:
        LOGGER.info("[vllm] health check skipped for dry-run handle")
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=timeout) as response:
                if response.status == 200:
                    data = json.load(response)
                    return "data" in data
        except URLError:
            time.sleep(0.5)
    return False
