"""Launch training jobs locally, via torchrun, or schedule Slurm sweeps."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

from nexa_compute.core.artifacts import ArtifactMeta  # type: ignore
from nexa_compute.training.hf_runner import HFTrainingConfig, cli as hf_cli, run_training
from nexa_infra.slurm import SlurmBatchArtifacts, prepare_slurm_batch
from nexa_train.train import run_training_job

from datetime import datetime, timezone


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def launch_training_job(
    config_path: Path,
    *,
    distributed: bool = False,
    overrides: Optional[list[str]] = None,
) -> ArtifactMeta:
    if distributed:
        cmd = [
            "bash",
            "scripts/launch_ddp.sh",
            "--config",
            str(config_path),
        ]
        if overrides:
            for override in overrides:
                cmd.extend(["--override", override])
        print("[nexa-infra] Launching distributed job:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return ArtifactMeta(
            kind="run",
            uri=str(config_path),
            hash="sha256:0",
            bytes=0,
            created_at=_now_utc(),
            inputs=[str(config_path)],
            labels={"note": "distributed_launch"},
        )
    print(f"[nexa-infra] Launching single-process training job with {config_path}")
    artifact = run_training_job(config_path, overrides=overrides or [])
    print(f"[nexa-infra] Training artifact COMPLETE -> {artifact.uri}")
    return artifact


def launch_hf_job(config: HFTrainingConfig | Sequence[str]) -> dict:
    """Launch the Hugging Face runner either from config or CLI args."""

    if isinstance(config, HFTrainingConfig):
        return run_training(config)
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        return hf_cli(list(config))
    raise TypeError("config must be HFTrainingConfig or iterable of CLI args")


def launch_slurm_sweep(config: Path, *, submit: bool = False) -> SlurmBatchArtifacts:
    """Generate Slurm sweep artifacts and optionally submit the job array."""

    artifacts = prepare_slurm_batch(config, submit=submit)
    print(f"[nexa-infra] Slurm script: {artifacts.script_path}")
    print(f"[nexa-infra] Slurm spec: {artifacts.spec_path}")
    print(f"[nexa-infra] Array size: {artifacts.job_count}")
    return artifacts
