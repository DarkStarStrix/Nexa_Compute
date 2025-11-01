#!/usr/bin/env python3
"""CLI entrypoint for launching the Hugging Face trainer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def setup_paths() -> None:
    """Ensure the project source directory is importable."""

if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    """Run the shared Hugging Face training CLI."""

    setup_paths()
    from nexa_compute.training.hf_runner import cli

    cli()


if __name__ == "__main__":
    main()
