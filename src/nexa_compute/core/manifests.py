"""Run Manifest System (V4 Compliance)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    """V4-compliant run manifest schema."""
    run_id: str
    timestamps: Dict[str, str] = Field(default_factory=lambda: {
        "start": datetime.now(timezone.utc).isoformat(),
        "end": None
    })
    monorepo_commit: str = "unknown"
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    dataset_version: Optional[str] = None
    hardware_specs: Dict[str, Any] = Field(default_factory=dict)
    shard_list: List[str] = Field(default_factory=list)
    tokens_processed: int = 0
    metrics: Dict[str, float] = Field(default_factory=dict)
    exit_status: str = "running"  # running, completed, failed, cancelled
    resume_info: Optional[Dict[str, Any]] = None

    def save(self, runs_dir: Path) -> Path:
        """Save manifest to disk."""
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{self.run_id}.json"
        with path.open("w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Load manifest from disk."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def update_status(self, status: str, metrics: Optional[Dict[str, float]] = None) -> None:
        """Update run status and metrics."""
        self.exit_status = status
        if status in ["completed", "failed", "cancelled"]:
            self.timestamps["end"] = datetime.now(timezone.utc).isoformat()
        if metrics:
            self.metrics.update(metrics)


def get_git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
