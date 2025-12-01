"""Real-time model performance monitoring and drift detection."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from nexa_compute.monitoring.alerts import AlertSeverity, send_alert
from nexa_compute.monitoring.drift import DriftDetector

LOGGER = logging.getLogger(__name__)


@dataclass
class MetricWindow:
    """Sliding window of metric values."""
    window_size: int = 1000
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, value: float) -> None:
        self.values.append(value)
        self.timestamps.append(time.time())


class ModelMonitor:
    """Monitors model inputs, outputs, and performance metrics."""

    def __init__(self, model_name: str, version: str) -> None:
        self.model_name = model_name
        self.version = version
        self.drift_detector = DriftDetector(threshold=0.05)
        
        # Feature distributions
        self.reference_distributions: Dict[str, List[float]] = {}
        self.current_distributions: Dict[str, MetricWindow] = {}
        
        # Performance metrics
        self.metrics: Dict[str, MetricWindow] = {}

    def set_reference(self, feature_name: str, data: List[float]) -> None:
        """Set baseline distribution for a feature."""
        self.reference_distributions[feature_name] = data

    def log_inference(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Log single inference event."""
        # Track numeric inputs for drift
        for key, value in inputs.items():
            if isinstance(value, (int, float)):
                if key not in self.current_distributions:
                    self.current_distributions[key] = MetricWindow()
                self.current_distributions[key].add(float(value))

        # Track latency
        if "latency" not in self.metrics:
            self.metrics["latency"] = MetricWindow()
        self.metrics["latency"].add(latency_ms)

    def check_drift(self) -> Dict[str, Any]:
        """Run drift detection on all tracked features."""
        results = {}
        
        for feature, ref_data in self.reference_distributions.items():
            if feature not in self.current_distributions:
                continue
                
            current_data = list(self.current_distributions[feature].values)
            if len(current_data) < 50: # Minimum sample size
                continue
                
            drift_result = self.drift_detector.detect_drift(ref_data, current_data)
            results[feature] = drift_result
            
            if drift_result["drift_detected"]:
                self._alert_drift(feature, drift_result)
                
        return results

    def _alert_drift(self, feature: str, result: Dict[str, Any]) -> None:
        send_alert(
            title=f"Data Drift Detected: {self.model_name} v{self.version}",
            message=f"Feature '{feature}' shows significant drift (p={result.get('p_value', 'N/A')})",
            severity=AlertSeverity.WARNING,
            source="model_monitor",
            metadata={
                "model": self.model_name,
                "version": self.version,
                "feature": feature,
                "statistic": result["statistic"],
            },
        )

