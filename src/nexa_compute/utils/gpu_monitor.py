"""Lightweight NVML-based GPU telemetry helper."""

from __future__ import annotations

import time
from typing import Iterable

import pynvml


def stream_gpu_stats(interval: float = 5.0, devices: Iterable[int] | None = None) -> None:
    """Continuously print GPU utilisation and memory usage.

    Args:
        interval: Seconds to sleep between samples.
        devices: Optional iterable of device indices. Defaults to all GPUs.
    """

    pynvml.nvmlInit()
    try:
        device_indices = list(devices) if devices is not None else list(range(pynvml.nvmlDeviceGetCount()))
        while True:
            lines = []
            for idx in device_indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                lines.append(
                    f"GPU{idx} util={util.gpu:3d}% mem={util.memory:3d}% "
                    f"({memory.used / (1024 ** 3):.2f}/{memory.total / (1024 ** 3):.2f} GB)"
                )
            print(" | ".join(lines))
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()
