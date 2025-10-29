"""Typer-based CLI for Nexa Compute workflows."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import List, Optional

import torch
import typer

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nexa_compute.config import load_config  # type: ignore  # noqa: E402
from nexa_compute.config.schema import TrainingConfig  # type: ignore  # noqa: E402
from nexa_compute.data import DataPipeline  # type: ignore  # noqa: E402
from nexa_compute.evaluation import Evaluator  # type: ignore  # noqa: E402
from nexa_compute.models import DEFAULT_MODEL_REGISTRY  # type: ignore  # noqa: E402
from nexa_compute.orchestration import TrainingPipeline  # type: ignore  # noqa: E402
from nexa_compute.utils.checkpoint import load_checkpoint, save_checkpoint  # type: ignore  # noqa: E402

app = typer.Typer(help="Nexa Compute orchestration CLI")


def _load_training_config(config_path: Path, overrides: Optional[List[str]]) -> TrainingConfig:
    return load_config(config_path, overrides=overrides or [])


@app.command()
def prepare_data(
    config: Path = typer.Option(..., exists=True, help="Path to YAML config"),
    override: Optional[List[str]] = typer.Option(None, "--override", help="Override key=value pairs"),
) -> None:
    cfg = _load_training_config(config, override)
    pipeline = DataPipeline(cfg.data)
    metadata_path = pipeline.materialize_metadata(cfg.output_directory())
    typer.echo(f"Data metadata saved to {metadata_path}")


@app.command()
def train(
    config: Path = typer.Option(..., exists=True, help="Path to YAML config"),
    override: Optional[List[str]] = typer.Option(None, "--override", help="Override key=value pairs"),
    disable_eval: bool = typer.Option(False, help="Skip evaluation after training"),
) -> None:
    pipeline = TrainingPipeline.from_config_file(config, overrides=override or [])
    artifacts = pipeline.run(enable_evaluation=not disable_eval)
    typer.echo(f"Run directory: {artifacts.run_dir}")
    if artifacts.checkpoint:
        typer.echo(f"Checkpoint: {artifacts.checkpoint}")
    if artifacts.metrics:
        typer.echo(f"Metrics: {json.dumps(artifacts.metrics, indent=2)}")


@app.command()
def evaluate(
    config: Path = typer.Option(..., exists=True, help="Path to YAML config"),
    checkpoint: Optional[Path] = typer.Option(None, help="Path to checkpoint (.pt)"),
    override: Optional[List[str]] = typer.Option(None, "--override", help="Override key=value pairs"),
) -> None:
    cfg = _load_training_config(config, override)
    pipeline = DataPipeline(cfg.data)
    dataloader = pipeline.dataloader("validation", batch_size=cfg.evaluation.batch_size)
    model = DEFAULT_MODEL_REGISTRY.build(cfg.model)
    if checkpoint:
        state = load_checkpoint(checkpoint)
        model.load_state_dict(state["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    evaluator = Evaluator(cfg.evaluation, device=device)
    metrics = evaluator.evaluate(model, dataloader)
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def tune(
    config: Path = typer.Option(..., exists=True, help="Path to YAML config"),
    max_trials: int = typer.Option(5, help="Number of random search trials"),
    override: Optional[List[str]] = typer.Option(None, "--override", help="Base overrides"),
) -> None:
    from .hyperparameter_search import random_search  # type: ignore

    cfg = _load_training_config(config, override)
    results = random_search(cfg, trials=max_trials)
    typer.echo(json.dumps(results, indent=2))


@app.command()
def package(
    config: Path = typer.Option(..., exists=True, help="Path to YAML config"),
    checkpoint: Optional[Path] = typer.Option(None, help="Checkpoint to package"),
    output: Path = typer.Option(Path("artifacts/package"), help="Output directory"),
    override: Optional[List[str]] = typer.Option(None, "--override", help="Override key=value pairs"),
) -> None:
    cfg = _load_training_config(config, override)
    run_dir = cfg.output_directory()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint
    if checkpoint_path is None:
        candidate = Path(cfg.training.checkpoint.dir) / "checkpoint_latest.pt"
        checkpoint_path = candidate if candidate.exists() else None
    if checkpoint_path and checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path)
        save_checkpoint({"model_state": state["model_state"]}, output, filename="model.pt")
    else:
        model = DEFAULT_MODEL_REGISTRY.build(cfg.model)
        save_checkpoint({"model_state": model.state_dict()}, output, filename="model.pt")
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(json.loads(cfg.model_dump_json()), handle, indent=2)
    archive_path = output / "package.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in output.iterdir():
            if item == archive_path:
                continue
            tar.add(item, arcname=item.name)
    typer.echo(f"Package created at {archive_path}")


if __name__ == "__main__":
    app()
