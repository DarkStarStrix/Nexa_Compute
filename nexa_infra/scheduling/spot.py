"""Utilities for managing Spot/Preemptible instances."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)


class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    UNKNOWN = "unknown"


@dataclass
class SpotInstanceInfo:
    provider: CloudProvider
    instance_id: str
    spot_request_id: Optional[str] = None
    region: str = "unknown"


class SpotManager:
    """Handles spot instance lifecycle and preemption."""

    def __init__(self) -> None:
        self.provider = self._detect_provider()
        self.metadata_url = self._get_metadata_url()

    def _detect_provider(self) -> CloudProvider:
        # Heuristic detection based on vendor-specific files or env vars
        # In production, this would be more robust
        try:
            with open("/sys/class/dmi/id/product_version", "r") as f:
                content = f.read().lower()
                if "amazon" in content:
                    return CloudProvider.AWS
                if "google" in content:
                    return CloudProvider.GCP
        except FileNotFoundError:
            pass
        return CloudProvider.UNKNOWN

    def _get_metadata_url(self) -> str:
        if self.provider == CloudProvider.AWS:
            return "http://169.254.169.254/latest/meta-data/"
        elif self.provider == CloudProvider.GCP:
            return "http://metadata.google.internal/computeMetadata/v1/"
        return ""

    def check_preemption(self) -> bool:
        """Check if the instance is marked for preemption."""
        if self.provider == CloudProvider.AWS:
            try:
                # AWS gives 2 minute warning via this endpoint
                resp = requests.get(
                    f"{self.metadata_url}spot/instance-action",
                    timeout=0.5,
                )
                return resp.status_code == 200
            except requests.RequestException:
                return False
                
        elif self.provider == CloudProvider.GCP:
            try:
                resp = requests.get(
                    f"{self.metadata_url}instance/preempted",
                    headers={"Metadata-Flavor": "Google"},
                    timeout=0.5,
                )
                return resp.text == "TRUE"
            except requests.RequestException:
                return False
                
        return False

    def get_instance_info(self) -> SpotInstanceInfo:
        # Implementation omitted for brevity
        return SpotInstanceInfo(self.provider, "i-unknown")

# Global instance
_SPOT_MANAGER = SpotManager()

def check_spot_interruption() -> bool:
    """Check if the current spot instance is about to be terminated."""
    return _SPOT_MANAGER.check_preemption()

