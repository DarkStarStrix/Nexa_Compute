#!/usr/bin/env python3
"""Cleanup utility for NexaCompute artifacts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_CHECKPOINT_ROOT = Path("/mnt/nexa_durable/checkpoints")
DEFAULT_ARCHIVE_ROOT = Path("/mnt/nexa_durable/archive")


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for cleanup actions."""

    parser = argparse.ArgumentParser(description="Prune or archive old training artifacts")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT, help="Durable checkpoint root")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT, help="Archive destination")
    parser.add_argument("--days", type=int, default=7, help="Archive checkpoints older than this many days")
    parser.add_argument("--prune-size", type=str, default="+2G", help="Delete files larger than this (find syntax)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    return parser.parse_args()


def prune_checkpoints(root: Path, size: str, dry_run: bool) -> None:
    """Remove large checkpoint files according to the configured size threshold."""

    cmd = [
        "find",
        str(root),
        "-type",
        "f",
        "-size",
        size,
        "-mtime",
        "+3",
    ]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd + ["-delete"]))
        return
    subprocess.run(cmd + ["-delete"], check=False)


def archive_checkpoints(root: Path, archive_root: Path, days: int, dry_run: bool) -> None:
    """Archive checkpoint directories older than the specified retention window."""

    cmd = [
        "find",
        str(root),
        "-maxdepth",
        "1",
        "-type",
        "d",
        "-mtime",
        f"+{days}",
    ]
    if dry_run:
        subprocess.run(cmd, check=False)
        return

    archive_root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    for line in result.stdout.strip().splitlines():
        path = Path(line)
        if path == root:
            continue
        archive_path = archive_root / f"{path.name}.tar.gz"
        subprocess.run(["tar", "-czf", str(archive_path), "-C", str(path), "."], check=False)
        subprocess.run(["rm", "-rf", str(path)], check=False)
        print(f"Archived {path} -> {archive_path}")


def main() -> None:
    """Run pruning and archiving workflows based on CLI inputs."""

    args = parse_args()
    prune_checkpoints(args.checkpoint_root, args.prune_size, args.dry_run)
    archive_checkpoints(args.checkpoint_root, args.archive_root, args.days, args.dry_run)


if __name__ == "__main__":
    main()
