#!/usr/bin/env python3
"""Run NVML-based GPU utilisation monitor in the foreground."""

from __future__ import annotations

import argparse

from nexa_compute.utils.gpu_monitor import stream_gpu_stats


def parse_args() -> argparse.Namespace:
    """Return CLI options for GPU monitoring."""

    parser = argparse.ArgumentParser(description="Stream GPU utilisation via NVML")
    parser.add_argument("--interval", type=float, default=5.0, help="Sample interval in seconds")
    parser.add_argument("--devices", type=int, nargs="*", help="GPU indices to monitor (default: all)")
    return parser.parse_args()


def main() -> None:
    """Start streaming GPU statistics."""

    args = parse_args()
    stream_gpu_stats(interval=args.interval, devices=args.devices)


if __name__ == "__main__":
    main()
