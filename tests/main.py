"""Unified entrypoint for running the Nexa test suite."""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TARGET = pathlib.Path(__file__).resolve().parent

# Ensure we preemptively silence the noisy torch/pynvml warning before pytest starts
warnings.filterwarnings("ignore", message="The pynvml package is deprecated", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nexa tests via pytest")
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments to forward to pytest (default: run entire tests/ tree)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_args = args.pytest_args or [str(DEFAULT_TARGET)]
    print(f"[tests.main] Running pytest with args: {' '.join(target_args)}")
    return pytest.main(target_args)


if __name__ == "__main__":
    sys.exit(main())

