"""Teardown workflow for infrastructure resources."""

from __future__ import annotations

from pathlib import Path

from nexa_infra.utilities import ClusterConfig


def teardown_cluster(manifest: Path) -> None:
    """Remove recorded state for a provisioned cluster."""

    config = ClusterConfig.from_file(manifest)
    state_file = Path("runs/manifests") / f"cluster_{config.name}.json"
    if state_file.exists():
        state_file.unlink()
    print(f"[nexa-infra] Decommissioned cluster '{config.name}'")


__all__ = ["teardown_cluster"]


