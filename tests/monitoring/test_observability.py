"""Tests for monitoring and alerting."""

import logging
from dataclasses import dataclass
from typing import List

import pytest

from nexa_compute.monitoring.alerts import (
    Alert,
    AlertBackend,
    AlertManager,
    AlertSeverity,
    send_alert,
)
from nexa_compute.monitoring.drift import DriftDetector
from nexa_compute.utils.metrics import MetricsRegistry

@dataclass
class MockBackend(AlertBackend):
    alerts: List[Alert]

    def __init__(self):
        self.alerts = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)

def test_alert_routing():
    manager = AlertManager()
    # Remove default log backend
    manager._backends = []
    
    backend = MockBackend()
    manager.register_backend(backend)
    
    manager.alert(
        title="Test Alert",
        message="Something happened",
        severity=AlertSeverity.ERROR,
        source="test",
    )
    
    assert len(backend.alerts) == 1
    alert = backend.alerts[0]
    assert alert.title == "Test Alert"
    assert alert.severity == AlertSeverity.ERROR

def test_drift_detection():
    detector = DriftDetector(threshold=0.05)
    
    # No drift (same distribution)
    ref = [1.0, 2.0, 3.0, 4.0, 5.0] * 100
    curr = [1.0, 2.0, 3.0, 4.0, 5.0] * 100
    
    result = detector.detect_drift(ref, curr)
    assert not result["drift_detected"]
    
    # Significant drift
    curr_shifted = [x + 10.0 for x in ref]
    result = detector.detect_drift(ref, curr_shifted)
    assert result["drift_detected"]

def test_metrics_recording():
    # Just verify no exceptions are raised
    # Checking actual Prometheus output requires parsing text format
    MetricsRegistry.record_training_metrics(
        run_id="test-run",
        model_name="test-model",
        metrics={"loss": 0.5, "accuracy": 0.9},
    )
    MetricsRegistry.record_gpu_stats({"gpu_0_util": 90.0})
    MetricsRegistry.record_api_request("GET", "/test", 200, 0.1)

