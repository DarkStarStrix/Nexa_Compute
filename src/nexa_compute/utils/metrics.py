"""Prometheus metrics export utilities.

This module provides a centralized registry for Prometheus metrics
and helper functions to record standard metrics.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, REGISTRY, start_http_server

# --- Training Metrics ---

TRAINING_LOSS = Gauge(
    "nexa_training_loss",
    "Current training loss",
    ["run_id", "model_name", "phase"],  # phase: train, validation
)

TRAINING_ACCURACY = Gauge(
    "nexa_training_accuracy",
    "Current training accuracy",
    ["run_id", "model_name", "phase"],
)

TRAINING_EPOCH = Gauge(
    "nexa_training_epoch",
    "Current training epoch",
    ["run_id", "model_name"],
)

GPU_UTILIZATION = Gauge(
    "nexa_gpu_utilization",
    "GPU utilization percentage",
    ["device_id", "node_id"],
)

GPU_MEMORY_USED = Gauge(
    "nexa_gpu_memory_used_mb",
    "GPU memory used in MB",
    ["device_id", "node_id"],
)

# --- API Metrics ---

API_REQUESTS_TOTAL = Counter(
    "nexa_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)

API_REQUEST_LATENCY = Histogram(
    "nexa_api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
)

# --- Infrastructure Metrics ---

JOB_QUEUE_DEPTH = Gauge(
    "nexa_job_queue_depth",
    "Number of jobs in the queue",
    ["queue_name"],
)

ACTIVE_WORKERS = Gauge(
    "nexa_active_workers",
    "Number of active workers",
    ["pool_name"],
)

# --- Cost Metrics ---

ESTIMATED_COST = Counter(
    "nexa_estimated_cost_usd",
    "Estimated accumulated cost in USD",
    ["run_id", "project_id"],
)


class MetricsRegistry:
    """Central registry for application metrics."""

    @staticmethod
    def record_training_metrics(
        run_id: str,
        model_name: str,
        metrics: Dict[str, float],
        phase: str = "train",
    ) -> None:
        """Record standard training metrics."""
        if "loss" in metrics:
            TRAINING_LOSS.labels(run_id=run_id, model_name=model_name, phase=phase).set(
                metrics["loss"]
            )
        if "accuracy" in metrics:
            TRAINING_ACCURACY.labels(run_id=run_id, model_name=model_name, phase=phase).set(
                metrics["accuracy"]
            )
        if "epoch" in metrics:
            TRAINING_EPOCH.labels(run_id=run_id, model_name=model_name).set(metrics["epoch"])

    @staticmethod
    def record_gpu_stats(stats: Dict[str, float], node_id: str = "local") -> None:
        """Record GPU utilization statistics."""
        # Expected format: {"gpu_0_util": 85.0, "gpu_0_mem": 4096.0}
        for key, value in stats.items():
            parts = key.split("_")
            if len(parts) >= 3 and parts[0] == "gpu":
                device_id = parts[1]
                metric_type = parts[2]  # util or mem
                if metric_type.startswith("util"):
                    GPU_UTILIZATION.labels(device_id=device_id, node_id=node_id).set(value)
                elif metric_type.startswith("mem"):
                    GPU_MEMORY_USED.labels(device_id=device_id, node_id=node_id).set(value)

    @staticmethod
    def record_api_request(method: str, endpoint: str, status_code: int, latency: float) -> None:
        """Record API request metrics."""
        API_REQUESTS_TOTAL.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()
        API_REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    @staticmethod
    def start_server(port: int = 9090) -> None:
        """Start a standalone Prometheus metrics server."""
        start_http_server(port)

