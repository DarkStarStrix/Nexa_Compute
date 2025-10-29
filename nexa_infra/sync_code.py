"""Code synchronisation helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def sync_repository(destination: str, *, include_runs: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    paths = ["*.py", "nexa_*", "src", "scripts", "pyproject.toml", "requirements.txt", "orchestrate.py"]
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
