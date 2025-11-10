"""Local process scheduler backend."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

LOGGER = logging.getLogger(__name__)

__all__ = ["LocalScheduler", "LocalCommandResult"]


@dataclass(frozen=True)
class LocalCommandResult:
    """Result returned by :class:`LocalScheduler`."""

    returncode: int
    stdout: str | None
    stderr: str | None


class LocalScheduler:
    """Run commands locally with optional environment overrides."""

    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture_output: bool = True,
        check: bool = True,
    ) -> LocalCommandResult:
        env_vars = os.environ.copy()
        if env:
            env_vars.update({str(key): str(value) for key, value in env.items()})
        LOGGER.info("[local-scheduler] running command: %s", " ".join(command))
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=env_vars,
            capture_output=capture_output,
            text=True,
            check=False,
        )
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return LocalCommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout if capture_output else None,
            stderr=completed.stderr if capture_output else None,
        )

    def spawn(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> subprocess.Popen:
        env_vars = os.environ.copy()
        if env:
            env_vars.update({str(key): str(value) for key, value in env.items()})
        LOGGER.info("[local-scheduler] spawning command: %s", " ".join(command))
        return subprocess.Popen(list(command), cwd=str(cwd) if cwd else None, env=env_vars)
