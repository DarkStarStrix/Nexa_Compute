"""Provisioning workflow for Nexa Compute clusters."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .slurm import prepare_slurm_batch
from .utils import ClusterConfig, run_command


def provision_cluster(manifest: Path, bootstrap_tailscale: bool = False) -> None:
    config = ClusterConfig.from_file(manifest)
    print(f"[nexa-infra] Provisioning cluster '{config.name}' in {config.region} using {config.provider}")
    # Placeholder for IaaS provisioning (Terraform/Pulumi hooks can go here)
    state_dir = Path("runs/manifests")
    state_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(state_dir / f"cluster_{config.name}.json")
    if bootstrap_tailscale:
        _bootstrap_tailscale()
    print("[nexa-infra] Cluster manifest recorded")


def create_slurm_sweep(config: Path, submit: bool = False) -> None:
    """Materialize Slurm sweep artifacts from a YAML description."""

    artifacts = prepare_slurm_batch(config, submit=submit)
    print(f"[nexa-infra] Slurm spec: {artifacts.spec_path}")
    print(f"[nexa-infra] Slurm script: {artifacts.script_path}")
    if artifacts.cost_manifest is not None:
        print(f"[nexa-infra] Cost estimate stored at {artifacts.cost_manifest}")


def _bootstrap_tailscale() -> None:
    script = Path(__file__).parent / "tailscale_bootstrap.sh"
    if not script.exists():
        raise FileNotFoundError("tailscale_bootstrap.sh missing")
    print("[nexa-infra] Bootstrapping Tailscale mesh...")
    run_command(["bash", str(script)])
