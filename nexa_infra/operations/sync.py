"""Code synchronisation helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def sync_repository(destination: str, *, include_runs: bool = False) -> None:
    """Sync local project code to a remote destination using rsync."""

    root = Path(__file__).resolve().parents[2]
    paths = [
        "*.py",
        "nexa_*",
        "src",
        "scripts",
        "pyproject.toml",
        "requirements.txt",
        "orchestrate.py",
    ]
    if include_runs:
        paths.append("runs")
    rsync_args = [
        "rsync",
        "-avz",
        "--delete",
    ]
    rsync_args.extend(paths)
    rsync_args.append(destination)
    print("[nexa-infra] syncing:", " ".join(paths))
    subprocess.run(" ".join(rsync_args), shell=True, check=True)


__all__ = ["sync_repository"]


