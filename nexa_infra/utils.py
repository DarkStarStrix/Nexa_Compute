"""Utility helpers for infrastructure workflows."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ClusterConfig:
    name: str
    provider: str
    region: str
    node_group: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def from_file(cls, path: Path) -> "ClusterConfig":
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(
            name=payload["name"],
            provider=payload["provider"],
            region=payload["region"],
            node_group=payload["node_group"],
            metadata=payload.get("metadata", {}),
        )

    def to_json(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "name": self.name,
                    "provider": self.provider,
                    "region": self.region,
                    "node_group": self.node_group,
                    "metadata": self.metadata,
                },
                handle,
                indent=2,
            )


def run_command(cmd: list[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a shell command with optional cwd."""
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=False)
