"""Utilities for building and launching Slurm job arrays for NexaCompute."""

from __future__ import annotations

import itertools
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .cost_tracker import estimate_batch_cost


@dataclass
class SlurmLauncher:
    """Configuration for how each task should be launched."""

    type: str = "python"  # "python" | "torchrun"
    script: str = "scripts/test_hf_train.py"
    python: str = "python3"
    module: bool = False
    entrypoint: Optional[str] = None  # used when module=True or torchrun module
    nproc_per_node: Optional[int] = None
    nnodes: Optional[int] = None
    rdzv_backend: str = "c10d"
    rdzv_endpoint: Optional[str] = None
    rdzv_id: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)


@dataclass
class SlurmJob:
    """High-level Slurm batch configuration."""

    name: str
    partition: str
    time: str
    nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 4
    ntasks_per_node: int = 1
    qos: Optional[str] = None
    account: Optional[str] = None
    constraint: Optional[str] = None
    gres: Optional[str] = None
    memory: Optional[str] = None
    output_dir: str = "logs/slurm"
    mail_type: Optional[str] = None
    mail_user: Optional[str] = None
    array_parallelism: Optional[int] = None
    modules: List[str] = field(default_factory=list)
    pre_commands: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    gpu_type: Optional[str] = None
    cost_per_gpu_hour: Optional[float] = None


@dataclass
class SweepDefinition:
    base_args: List[str] = field(default_factory=list)
    parameters: Dict[str, List[Any]] = field(default_factory=dict)
    overrides: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    wandb_group: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class SlurmBatchArtifacts:
    script_path: Path
    spec_path: Path
    job_count: int
    commands: List[List[str]]
    cost_manifest: Optional[Path]


def _parse_time_to_hours(time_str: str) -> float:
    parts = time_str.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid time format '{time_str}', expected HH:MM[:SS]")
    if len(parts) == 2:
        hours, minutes = parts
        seconds = 0
    else:
        hours, minutes, seconds = parts
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return total_seconds / 3600.0


def _ensure_absolute(path: Path) -> Path:
    return path if path.is_absolute() else path.resolve()


def _expand_parameter_grid(parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not parameters:
        return [{}]
    keys = list(parameters.keys())
    values_product = itertools.product(*(parameters[key] for key in keys))
    combos = []
    for values in values_product:
        combos.append({key: value for key, value in zip(keys, values)})
    return combos


def _flag_name(key: str) -> str:
    normalized = key.replace("_", "-")
    return f"--{normalized}" if not normalized.startswith("-") else normalized


def _build_command(
    launcher: SlurmLauncher,
    job: SlurmJob,
    args: List[str],
) -> List[str]:
    if launcher.type not in {"python", "torchrun"}:
        raise ValueError(f"Unsupported launcher type '{launcher.type}'")

    if launcher.type == "torchrun":
        nproc = launcher.nproc_per_node or job.gpus_per_node
        nnodes = launcher.nnodes or job.nodes
        cmd: List[str] = ["torchrun", f"--nnodes={nnodes}", f"--nproc_per_node={nproc}"]
        if launcher.rdzv_endpoint:
            cmd.append(f"--rdzv_endpoint={launcher.rdzv_endpoint}")
        if launcher.rdzv_backend:
            cmd.append(f"--rdzv_backend={launcher.rdzv_backend}")
        if launcher.rdzv_id:
            cmd.append(f"--rdzv_id={launcher.rdzv_id}")
        cmd.extend(launcher.extra_args)
        entrypoint = launcher.entrypoint or launcher.script
        if launcher.module:
            cmd.extend(["-m", entrypoint])
        else:
            cmd.append(entrypoint)
        cmd.extend(args)
        return cmd

    # Default python launcher
    cmd = [launcher.python, launcher.script]
    cmd.extend(launcher.extra_args)
    cmd.extend(args)
    return cmd


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_to_list(value: Optional[Iterable[Any]]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def prepare_slurm_batch(
    config_path: Path,
    *,
    submit: bool = False,
    output_dir: Optional[Path] = None,
) -> SlurmBatchArtifacts:
    """Create Slurm batch script and command spec from sweep configuration."""

    payload = _load_yaml(config_path)

    job_cfg = payload.get("job") or {}
    sweep_cfg = payload.get("sweep") or {}
    launcher_cfg = payload.get("launcher") or {}

    job = SlurmJob(
        name=job_cfg.get("name", "nexa-train"),
        partition=job_cfg.get("partition", "gpu"),
        time=job_cfg.get("time", "08:00:00"),
        nodes=int(job_cfg.get("nodes", 1)),
        gpus_per_node=int(job_cfg.get("gpus_per_node", 1)),
        cpus_per_task=int(job_cfg.get("cpus_per_task", 6)),
        ntasks_per_node=int(job_cfg.get("ntasks_per_node", 1)),
        qos=job_cfg.get("qos"),
        account=job_cfg.get("account"),
        constraint=job_cfg.get("constraint"),
        gres=job_cfg.get("gres"),
        memory=job_cfg.get("memory"),
        output_dir=job_cfg.get("output_dir", "logs/slurm"),
        mail_type=job_cfg.get("mail_type"),
        mail_user=job_cfg.get("mail_user"),
        array_parallelism=job_cfg.get("array_parallelism"),
        modules=_normalize_to_list(job_cfg.get("modules")),
        pre_commands=_normalize_to_list(job_cfg.get("pre_commands")),
        env=job_cfg.get("env", {}),
        gpu_type=job_cfg.get("gpu_type"),
        cost_per_gpu_hour=job_cfg.get("cost_per_gpu_hour"),
    )

    launcher = SlurmLauncher(
        type=launcher_cfg.get("type", "python"),
        script=launcher_cfg.get("script", "scripts/test_hf_train.py"),
        python=launcher_cfg.get("python", "python3"),
        module=bool(launcher_cfg.get("module", False)),
        entrypoint=launcher_cfg.get("entrypoint"),
        nproc_per_node=launcher_cfg.get("nproc_per_node"),
        nnodes=launcher_cfg.get("nnodes"),
        rdzv_backend=launcher_cfg.get("rdzv_backend", "c10d"),
        rdzv_endpoint=launcher_cfg.get("rdzv_endpoint"),
        rdzv_id=launcher_cfg.get("rdzv_id"),
        extra_args=_normalize_to_list(launcher_cfg.get("extra_args")),
    )

    sweep = SweepDefinition(
        base_args=_normalize_to_list(sweep_cfg.get("base_args")),
        parameters=sweep_cfg.get("parameters", {}),
        overrides=_normalize_to_list(sweep_cfg.get("overrides")),
        tags=_normalize_to_list(sweep_cfg.get("tags")),
        wandb_group=sweep_cfg.get("wandb_group"),
        env=sweep_cfg.get("env", {}),
    )

    combos = _expand_parameter_grid(sweep.parameters)
    job_count = len(combos)
    if job_count == 0:
        raise ValueError("Sweep produced no jobs; check parameter definitions")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = (
        output_dir.resolve()
        if output_dir is not None
        else Path("runs/manifests/slurm").resolve()
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_subdir = batch_dir / f"{job.name}_{timestamp}"
    batch_subdir.mkdir(parents=True, exist_ok=True)

    spec_path = batch_subdir / "spec.json"

    commands: List[List[str]] = []
    spec_payload: List[Dict[str, Any]] = []

    for index, assignment in enumerate(combos):
        args = list(sweep.base_args)
        args.extend(sweep.overrides)
        tags = list(sweep.tags)

        for key, value in assignment.items():
            flag = _flag_name(key)
            tags.append(f"{key}:{value}")
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])

        command = _build_command(launcher, job, args)
        commands.append(command)

        env_vars = {**job.env, **sweep.env}
        env_vars.setdefault("WANDB_RUN_GROUP", sweep.wandb_group or job.name)
        env_vars.setdefault("WANDB_RUN_NAME", f"{job.name}_{index:03d}")

        spec_payload.append(
            {
                "command": command,
                "env": env_vars,
                "tags": tags,
                "assignment": assignment,
            }
        )

    spec_path.write_text(json.dumps(spec_payload, indent=2))

    script_path = batch_subdir / "job.sbatch"
    script_path.write_text(
        _render_slurm_script(
            job=job,
            job_count=job_count,
            spec_path=spec_path,
        ),
        encoding="utf-8",
    )

    cost_manifest: Optional[Path] = None
    try:
        walltime_hours = _parse_time_to_hours(job.time)
        cost_manifest = estimate_batch_cost(
            run_id=f"slurm_{job.name}_{timestamp}",
            nodes=job.nodes,
            gpus_per_node=job.gpus_per_node,
            duration_hours=walltime_hours,
            gpu_type=job.gpu_type,
            gpu_hour_cost=job.cost_per_gpu_hour,
            output_dir=batch_subdir,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[slurm] Cost estimation skipped: {exc}")

    if submit:
        subprocess.run(["sbatch", str(script_path)], check=True)

    return SlurmBatchArtifacts(
        script_path=script_path,
        spec_path=spec_path,
        job_count=job_count,
        commands=commands,
        cost_manifest=cost_manifest,
    )


def _render_slurm_script(*, job: SlurmJob, job_count: int, spec_path: Path) -> str:
    lines: List[str] = ["#!/usr/bin/env bash"]
    lines.append(f"#SBATCH --job-name={job.name}")
    lines.append(f"#SBATCH --partition={job.partition}")
    lines.append(f"#SBATCH --nodes={job.nodes}")
    lines.append(f"#SBATCH --ntasks-per-node={job.ntasks_per_node}")
    lines.append(f"#SBATCH --cpus-per-task={job.cpus_per_task}")
    lines.append(f"#SBATCH --gpus-per-node={job.gpus_per_node}")
    lines.append(f"#SBATCH --time={job.time}")
    output_dir = Path(job.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lines.append(f"#SBATCH --output={output_dir}/%x-%A_%a.out")
    if job.account:
        lines.append(f"#SBATCH --account={job.account}")
    if job.qos:
        lines.append(f"#SBATCH --qos={job.qos}")
    if job.constraint:
        lines.append(f"#SBATCH --constraint={job.constraint}")
    if job.gres:
        lines.append(f"#SBATCH --gres={job.gres}")
    if job.memory:
        lines.append(f"#SBATCH --mem={job.memory}")
    if job.mail_type:
        lines.append(f"#SBATCH --mail-type={job.mail_type}")
    if job.mail_user:
        lines.append(f"#SBATCH --mail-user={job.mail_user}")

    array_line = f"#SBATCH --array=0-{job_count - 1}"
    if job.array_parallelism:
        array_line += f"%{job.array_parallelism}"
    lines.append(array_line)

    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("export JOB_SPEC=\"" + str(_ensure_absolute(spec_path)) + "\"")
    lines.append("export PYTHONUNBUFFERED=1")

    for module in job.modules:
        lines.append(f"module load {module}")

    for command in job.pre_commands:
        lines.append(command)

    lines.append("")
    lines.append("python3 - <<'PY'")
    lines.append("import json")
    lines.append("import os")
    lines.append("import shlex")
    lines.append("import subprocess")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("specs = json.loads(Path(os.environ['JOB_SPEC']).read_text())")
    lines.append("task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))")
    lines.append("if task_id < 0 or task_id >= len(specs):")
    lines.append("    raise SystemExit(f'Invalid SLURM_ARRAY_TASK_ID {task_id}')")
    lines.append("spec = specs[task_id]")
    lines.append("env = dict(os.environ)")
    lines.append("env.update(spec.get('env', {}))")
    lines.append("cmd = spec['command']")
    lines.append("print('[slurm] launching:', ' '.join(shlex.quote(token) for token in cmd))")
    lines.append("subprocess.run(cmd, check=True, env=env)")
    lines.append("PY")

    return "\n".join(lines) + "\n"


__all__ = [
    "prepare_slurm_batch",
    "SlurmBatchArtifacts",
]


