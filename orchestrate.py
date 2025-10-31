"""Top-level orchestration CLI for Nexa Compute platform."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pynvml
import torch
import typer

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = typer.Typer(help="Nexa Compute control surface")


@app.command()
def provision(cluster_config: Path = typer.Option(Path("nexa_infra/cluster.yaml"), exists=True), bootstrap: bool = typer.Option(False)) -> None:
    """Provision infrastructure resources defined in the cluster manifest."""
    from nexa_infra.provision import provision_cluster

    provision_cluster(cluster_config, bootstrap)


@app.command()
def sync(destination: str = typer.Argument(..., help="user@host:/path target"), include_runs: bool = typer.Option(False, help="Sync runs directory")) -> None:
    """Sync local project code to a remote host using rsync."""
    from nexa_infra.sync_code import sync_repository

    sync_repository(destination, include_runs=include_runs)


@app.command()
def launch(config: Path = typer.Option(Path("nexa_train/configs/baseline.yaml"), exists=True), distributed: bool = typer.Option(False, help="Respect distributed settings")) -> None:
    """Launch a training job using the training controller."""
    from nexa_infra.launch_job import launch_training_job

    launch_training_job(config, distributed=distributed)


@app.command()
def teardown(cluster_config: Path = typer.Option(Path("nexa_infra/cluster.yaml"), exists=True)) -> None:
    """Tear down provisioned infrastructure."""
    from nexa_infra.teardown import teardown_cluster

    teardown_cluster(cluster_config)


@app.command()
def cost(report_dir: Path = typer.Option(Path("runs/manifests"), exists=False)) -> None:
    """Summarise cost reports captured during runs."""
    from nexa_infra.cost_tracker import summarize_costs

    summarize_costs(report_dir)


@app.command()
def prepare_data(config: Path = typer.Option(Path("nexa_train/configs/baseline.yaml"), exists=True), materialize_only: bool = typer.Option(False, help="Skip download/augmentation")) -> None:
    """Prepare datasets according to the config manifest."""
    from nexa_data.prepare import prepare_from_config

    prepare_from_config(config, materialize_only=materialize_only)


@app.command()
def evaluate(config: Path = typer.Option(Path("nexa_train/configs/baseline.yaml"), exists=True), checkpoint: Optional[Path] = typer.Option(None)) -> None:
    """Run evaluation workflow for a trained checkpoint."""
    from nexa_eval.analyze import evaluate_checkpoint

    evaluate_checkpoint(config, checkpoint)


@app.command()
def feedback(config: Path = typer.Option(Path("nexa_train/configs/baseline.yaml"), exists=True)) -> None:
    """Kick off model feedback loop analysis."""
    from nexa_feedback.feedback_loop import run_feedback_cycle

    run_feedback_cycle(config)


@app.command()
def leaderboard(host: Optional[str] = typer.Option(None, help="Hostname to bind UI"), port: int = typer.Option(8080)) -> None:
    """Serve the experiment leaderboard UI."""
    from nexa_ui.leaderboard import serve_leaderboard

    serve_leaderboard(host or "0.0.0.0", port)


@app.command()
def summary(output: Path = typer.Option(Path("runs/manifests/infra_report.json"), help="Where to write infra report")) -> None:
    """Capture current infra fingerprint for reproducibility."""

    env_keys = [
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "TORCH_DISTRIBUTED_DEBUG",
        "OMP_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ]
    env_snapshot = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}

    gpu_info = []
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        try:
            for idx in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                props = torch.cuda.get_device_properties(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info.append(
                    {
                        "id": idx,
                        "name": props.name,
                        "memory_total_gb": round(props.total_memory / (1024 ** 3), 2),
                        "util_gpu_percent": util.gpu,
                        "util_mem_percent": util.memory,
                        "mem_used_gb": round(memory.used / (1024 ** 3), 3),
                    }
                )
        finally:
            pynvml.nvmlShutdown()

    manifests_dir = ROOT / "runs" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    wandb_runs = []
    if manifests_dir.exists():
        for manifest_path in manifests_dir.glob("*.json"):
            try:
                data = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:  # pragma: no cover
                continue
            wandb_id = data.get("wandb_run_id")
            if wandb_id:
                wandb_runs.append({"manifest": manifest_path.name, "wandb_run_id": wandb_id})

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version.split(" ")[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "env": env_snapshot,
        "gpu": gpu_info,
        "manifests_indexed": len(list(manifests_dir.glob("*.json"))),
        "wandb_runs": wandb_runs,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    typer.echo(f"Infra summary written to {output}")


if __name__ == "__main__":
    app()
