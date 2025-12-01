"""Provisioning and scaling infrastructure."""

from __future__ import annotations

import logging
from typing import List

from nexa_compute.infra.autoscaling import AutoScaler, AutoScalingConfig, ScalingPolicy

LOGGER = logging.getLogger(__name__)


class InfraScaler:
    """Interface to cloud provider scaling APIs."""

    def __init__(self, cluster_name: str) -> None:
        self.cluster_name = cluster_name
        self.autoscaler = AutoScaler(
            AutoScalingConfig(
                min_replicas=1,
                max_replicas=10,
                target_metric=ScalingPolicy.QUEUE_DEPTH,
                target_value=5.0, # 5 jobs per worker
            )
        )

    def update(self, metric_value: float) -> None:
        """Update cluster size based on metric."""
        desired_count = self.autoscaler.decide(metric_value)
        self._set_cluster_size(desired_count)

    def _set_cluster_size(self, count: int) -> None:
        # Placeholder for Terraform/AWS/GCP API call
        # e.g. aws autoscaling update-auto-scaling-group
        LOGGER.info(f"Setting cluster {self.cluster_name} size to {count}")

