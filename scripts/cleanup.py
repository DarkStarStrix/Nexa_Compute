#!/usr/bin/env python3
"""Cleanup utility for NexaCompute artifacts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_CHECKPOINT_ROOT = Path("/mnt/nexa_durable/checkpoints")
DEFAULT_ARCHIVE_ROOT = Path("/mnt/nexa_durable/archive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune or archive old training artifacts")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT, help="Durable checkpoint root")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT, help="Archive destination")
    parser.add_argument("--days", type=int, default=7, help="Archive checkpoints older than this many days")
    parser.add_argument("--prune-size", type=str, default="+2G", help="Delete files larger than this (find syntax)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prune_cmd = [
        "find",
        str(args.checkpoint_root),
        "-type",
        "f",
        "-size",
        args.prune_size,
        "-mtime",
        "+3",
    ]
    if not args.dry_run:
        subprocess.run(prune_cmd + ["-delete"], check=False)
    else:
        print("DRY-RUN:", " ".join(prune_cmd + ["-delete"]))

    archive_cmd = [
        "find",
        str(args.checkpoint_root),
        "-maxdepth",
        "1",
        "-type",
        "d",
        "-mtime",
        f"+{args.days}",
    ]
    if args.dry_run:
        subprocess.run(archive_cmd, check=False)
        return

    args.archive_root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(archive_cmd, capture_output=True, text=True, check=False)
    for line in result.stdout.strip().splitlines():
        path = Path(line)
        if path == args.checkpoint_root:
            continue
        archive_path = args.archive_root / f"{path.name}.tar.gz"
        subprocess.run(["tar", "-czf", str(archive_path), "-C", str(path), "."], check=False)
        subprocess.run(["rm", "-rf", str(path)], check=False)
        print(f"Archived {path} -> {archive_path}")


if __name__ == "__main__":
    main()
