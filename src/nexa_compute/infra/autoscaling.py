"""Auto-scaling logic for infrastructure resources."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class ScalingPolicy(str, Enum):
    QUEUE_DEPTH = "queue_depth"
    CPU_UTILIZATION = "cpu_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    SCHEDULE = "schedule"


@dataclass
class AutoScalingConfig:
    min_replicas: int
    max_replicas: int
    target_metric: ScalingPolicy
    target_value: float
    scale_up_cooldown: int = 300
    scale_down_cooldown: int = 600


class AutoScaler:
    """Manages scaling decisions based on metrics."""

    def __init__(self, config: AutoScalingConfig) -> None:
        self.config = config
        self.current_replicas = config.min_replicas
        self.last_scale_time = 0.0

    def decide(self, current_metric_value: float) -> int:
        """Calculate desired replica count."""
        now = time.time()
        
        # Check cooldown
        if now - self.last_scale_time < self.config.scale_up_cooldown:
            return self.current_replicas

        desired = self.current_replicas
        
        # Simple proportional scaling logic
        # In production, use PID controller or more sophisticated logic
        if current_metric_value > self.config.target_value:
            # Scale up
            ratio = current_metric_value / self.config.target_value
            desired = int(self.current_replicas * ratio)
            desired = min(desired, self.config.max_replicas)
            
        elif current_metric_value < (self.config.target_value * 0.7):
            # Scale down (conservative)
            if now - self.last_scale_time > self.config.scale_down_cooldown:
                desired = max(self.current_replicas - 1, self.config.min_replicas)

        if desired != self.current_replicas:
            LOGGER.info(
                "scaling_decision",
                extra={
                    "current": self.current_replicas,
                    "desired": desired,
                    "metric": current_metric_value,
                    "target": self.config.target_value,
                },
            )
            self.current_replicas = desired
            self.last_scale_time = now
            
        return desired

