"""Alerting utilities for critical system events."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

from ..utils.retry import RetryPolicy, retry_call

LOGGER = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    title: str
    message: str
    severity: AlertSeverity
    source: str
    metadata: Dict[str, Any]
    timestamp: str


class AlertBackend(ABC):
    @abstractmethod
    def send(self, alert: Alert) -> None:
        """Send an alert to the backend."""
        pass


class LogBackend(AlertBackend):
    """Simple backend that logs alerts."""

    def send(self, alert: Alert) -> None:
        LOGGER.log(
            logging.getLevelName(alert.severity.value),
            f"ALERT: {alert.title} - {alert.message}",
            extra={"alert": asdict(alert)},
        )


class SlackBackend(AlertBackend):
    """Backend that sends alerts to Slack via webhook."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url
        self._retry_policy = RetryPolicy(max_attempts=3, base_delay=1.0)

    def send(self, alert: Alert) -> None:
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#7b00ff",
        }
        
        payload = {
            "text": f"*{alert.severity.value}*: {alert.title}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#cccccc"),
                    "fields": [
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp, "short": True},
                        {"title": "Message", "value": alert.message, "short": False},
                    ],
                    "footer": "NexaCompute Alert System",
                }
            ],
        }
        
        # Add metadata as fields if present
        if alert.metadata:
            fields = payload["attachments"][0]["fields"] # type: ignore
            for k, v in alert.metadata.items():
                fields.append({"title": k, "value": str(v), "short": True})

        try:
            retry_call(
                requests.post,
                self.webhook_url,
                json=payload,
                policy=self._retry_policy,
            )
        except Exception as exc:
            LOGGER.error("failed_to_send_slack_alert", extra={"error": str(exc)})


class AlertManager:
    """Central manager for routing alerts."""

    def __init__(self) -> None:
        self._backends: List[AlertBackend] = [LogBackend()]
        
        # Configure Slack if webhook is present
        slack_url = os.getenv("SLACK_WEBHOOK_URL")
        if slack_url:
            self._backends.append(SlackBackend(slack_url))
            
    def register_backend(self, backend: AlertBackend) -> None:
        self._backends.append(backend)

    def alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        source: str = "nexa-compute",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        from datetime import datetime, timezone
        
        alert_obj = Alert(
            title=title,
            message=message,
            severity=severity,
            source=source,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        for backend in self._backends:
            try:
                backend.send(alert_obj)
            except Exception as exc:
                LOGGER.error(
                    "alert_backend_failed",
                    extra={"backend": type(backend).__name__, "error": str(exc)},
                )

# Global instance
_ALERT_MANAGER = AlertManager()

def get_alert_manager() -> AlertManager:
    return _ALERT_MANAGER

def send_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    source: str = "nexa-compute",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Convenience function to send an alert via the global manager."""
    get_alert_manager().alert(title, message, severity, source, metadata)

