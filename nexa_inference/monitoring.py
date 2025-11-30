"""Monitoring utilities for inference server."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

try:
    from prometheus_client import Counter, Gauge, Histogram
except ImportError:  # pragma: no cover
    Counter = Histogram = Gauge = None  # type: ignore[assignment]


if Counter:
    REQUEST_COUNT = Counter("nexa_requests_total", "Total requests", ["endpoint", "status"])
    EMBEDDING_LATENCY = Histogram("nexa_embedding_latency_seconds", "Embedding latency")
    SEARCH_LATENCY = Histogram("nexa_search_latency_seconds", "Search latency")
    GPU_UTILIZATION = Gauge("nexa_gpu_utilization_percent", "GPU utilization")
else:  # pragma: no cover - fallback when prometheus client missing
    REQUEST_COUNT = EMBEDDING_LATENCY = SEARCH_LATENCY = GPU_UTILIZATION = None


@contextmanager
def observe_latency(metric, endpoint: str) -> Generator[None, None, None]:
    if metric is None:
        yield
        return
    timer = metric.labels(endpoint).time() if hasattr(metric, "labels") else metric.time()  # type: ignore[attr-defined]
    with timer:
        yield


def record_request(endpoint: str, status: int) -> None:
    if REQUEST_COUNT:
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()

