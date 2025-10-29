"""Teardown workflow for infrastructure resources."""

from __future__ import annotations

from pathlib import Path

from .utils import ClusterConfig


def teardown_cluster(manifest: Path) -> None:
    config = ClusterConfig.from_file(manifest)
    state_file = Path("runs/manifests") / f"cluster_{config.name}.json"
    if state_file.exists():
        state_file.unlink()
    print(f"[nexa-infra] Decommissioned cluster '{config.name}'")
