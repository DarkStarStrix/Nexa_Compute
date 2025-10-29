"""Top-level orchestration CLI for Nexa Compute platform."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


if __name__ == "__main__":
    app()
