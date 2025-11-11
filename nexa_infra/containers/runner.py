"""Bootstrap and execute curated Nexa container images."""

from __future__ import annotations

import os
import subprocess
from typing import Mapping, Sequence

from .registry import BOOTSTRAP_SCRIPT, ContainerSpec, available_containers, get_container


def run_container(
    target: str,
    *,
    override_image: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    bootstrap_args: Sequence[str] | None = None,
) -> None:
    """Launch a container via the shared bootstrap script."""

    if not BOOTSTRAP_SCRIPT.exists():
        raise FileNotFoundError(f"Bootstrap script missing at {BOOTSTRAP_SCRIPT}")

    spec = get_container(target)

    env = os.environ.copy()
    env.setdefault("WORK", "/workspace")
    env.setdefault("NVME", "/mnt/nvme")
    env["SHM_SIZE"] = spec.shm_size
    if extra_env:
        env.update(extra_env)

    image = override_image or spec.image
    cmd = ["bash", str(BOOTSTRAP_SCRIPT), image]
    if bootstrap_args:
        cmd.extend(bootstrap_args)

    subprocess.run(cmd, check=True, env=env)


__all__ = ["run_container", "available_containers", "ContainerSpec"]


