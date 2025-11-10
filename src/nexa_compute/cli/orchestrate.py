"""Typer CLI entrypoint for NexaCompute orchestration (v2)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import urlparse

import typer
import yaml

from ..core.dag import PipelineExecutor, PipelineGraph, PipelineStep
from ..core.registry import DEFAULT_DB_PATH, promote as registry_promote, register as registry_register, resolve as registry_resolve
from ..runners.eval import EvalRunSpec, EvalRunner
from ..runners.serve import ServeHandle, ServeRunSpec, ServeRunner
from ..runners.train import TrainRunSpec, TrainRunner

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = typer.Typer(help="NexaCompute orchestration CLI")
pipeline_app = typer.Typer(help="Pipeline execution commands")
models_app = typer.Typer(help="Registry operations")
serve_app = typer.Typer(help="Serving helpers")

app.add_typer(pipeline_app, name="pipeline")
app.add_typer(models_app, name="models")
app.add_typer(serve_app, name="serve")

STATE_ROOT = Path(".nexa_state")


@dataclass
class RuntimeContext:
    train_runner: TrainRunner
    eval_runner: EvalRunner
    serve_runner: ServeRunner

    def __init__(self) -> None:
        self.train_runner = TrainRunner()
        self.eval_runner = EvalRunner()
        self.serve_runner = ServeRunner()


RUNTIME = RuntimeContext()
ACTIVE_SERVE_HANDLES: dict[int, ServeHandle] = {}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _to_local_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path or value).expanduser().resolve()
    raise ValueError(f"Only local filesystem paths are supported, got '{value}'")


def _first_output_path(step: PipelineStep) -> Path:
    if not step.outputs:
        raise ValueError(f"Step '{step.step_id}' does not declare outputs")
    _, value = next(iter(step.outputs.items()))
    return _to_local_path(value)


def _get_input(step: PipelineStep, key: str | None = None) -> Optional[str]:
    if not step.inputs:
        return None
    if key and key in step.inputs:
        return step.inputs[key]
    _, value = next(iter(step.inputs.items()))
    return value


def _execute_step(step: PipelineStep) -> None:
    backend = step.uses
    if backend == "runners.train":
        _run_train_step(step)
        return
    if backend == "runners.eval":
        _run_eval_step(step)
        return
    if backend == "runners.serve":
        _run_serve_step(step)
        return
    if backend == "core.registry.register":
        _run_registry_register(step)
        return
    if backend == "core.registry.promote":
        _run_registry_promote(step)
        return
    LOGGER.warning("[pipeline] Step '%s' uses '%s' which is not implemented; skipping", step.step_id, backend)


def _resolve_env(params: Mapping[str, Any]) -> Mapping[str, str]:
    env = params.get("env")
    if not env:
        return {}
    return {str(k): str(v) for k, v in dict(env).items()}


def _run_train_step(step: PipelineStep) -> TrainResult:
    artifact_path = _first_output_path(step)
    params = dict(step.params)
    spec = TrainRunSpec(
        backend=step.backend or params.get("backend", "hf"),
        scheduler=step.scheduler or params.get("scheduler", "local"),
        params=params,
        artifact_path=artifact_path,
        env=_resolve_env(params),
    )
    LOGGER.info("[pipeline] executing train step '%s'", step.step_id)
    return RUNTIME.train_runner.run(spec)


def _run_eval_step(step: PipelineStep) -> EvalResult:
    artifact_path = _first_output_path(step)
    params = dict(step.params)
    config_value = params.get("config_path") or _get_input(step, "config") or _get_input(step)
    if not config_value:
        raise ValueError(f"Evaluation step '{step.step_id}' requires a config path")
    checkpoint_value = params.get("checkpoint") or _get_input(step, "checkpoint")
    checkpoint_path = _to_local_path(checkpoint_value) if checkpoint_value else None
    spec = EvalRunSpec(
        config_path=_to_local_path(config_value),
        checkpoint_path=checkpoint_path,
        artifact_path=artifact_path,
    )
    LOGGER.info("[pipeline] executing eval step '%s'", step.step_id)
    return RUNTIME.eval_runner.run(spec)


def _run_serve_step(step: PipelineStep) -> ServeHandle:
    params = dict(step.params)
    model_value = params.get("model_path") or _get_input(step, "model") or _get_input(step)
    if not model_value:
        raise ValueError(f"Serve step '{step.step_id}' requires a model path")
    spec = ServeRunSpec(
        backend=step.backend or params.get("backend", "vllm"),
        model_path=_to_local_path(model_value),
        host=params.get("host", "0.0.0.0"),
        port=int(params.get("port", 8000)),
        tensor_parallel=params.get("tensor_parallel"),
        config_path=_to_local_path(params["config_path"]) if "config_path" in params else None,
        env=_resolve_env(params),
        dry_run=bool(params.get("dry_run", False)),
    )
    LOGGER.info("[pipeline] starting serve step '%s'", step.step_id)
    handle = RUNTIME.serve_runner.start(spec)
    ACTIVE_SERVE_HANDLES[spec.port] = handle
    return handle


def _run_registry_register(step: PipelineStep) -> str:
    params = dict(step.params)
    name = params.get("name") or _get_input(step, "name")
    uri = params.get("uri") or _get_input(step, "uri") or _get_input(step)
    meta_path = params.get("meta") or _get_input(step, "meta")
    if not all([name, uri, meta_path]):
        raise ValueError(f"Registry register step '{step.step_id}' requires name, uri, and meta")
    meta = json.loads(Path(meta_path).read_text())
    version = registry_register(str(name), str(uri), meta, db_path=DEFAULT_DB_PATH)
    LOGGER.info("[registry] registered %s version %s -> %s", name, version, uri)
    return version


def _run_registry_promote(step: PipelineStep) -> None:
    params = dict(step.params)
    name = params.get("name")
    version = params.get("version")
    tag = params.get("tag", "latest")
    if not all([name, version]):
        raise ValueError(f"Registry promote step '{step.step_id}' requires name and version")
    registry_promote(str(name), str(version), str(tag), db_path=DEFAULT_DB_PATH)
    LOGGER.info("[registry] promoted %s version %s to tag %s", name, version, tag)


def _load_pipeline_steps(pipeline_payload: Mapping[str, Any]) -> tuple[str, Iterable[PipelineStep]]:
    name = pipeline_payload.get("name", "unnamed")
    steps_payload = pipeline_payload.get("steps") or []
    steps = [PipelineStep.from_dict(step) for step in steps_payload]
    return name, steps


def _pipeline_state_dir(pipeline_name: str) -> Path:
    return STATE_ROOT / pipeline_name


@pipeline_app.command()
def run(pipeline_file: Path = typer.Argument(..., exists=True)) -> None:
    """Execute a pipeline specification."""

    payload = _load_yaml(pipeline_file)
    pipeline_payload = payload.get("pipeline")
    if not pipeline_payload:
        raise typer.BadParameter("Pipeline YAML must contain a 'pipeline' section")
    pipeline_name, steps = _load_pipeline_steps(pipeline_payload)
    state_dir = _pipeline_state_dir(pipeline_name)
    state_dir.mkdir(parents=True, exist_ok=True)
    graph = PipelineGraph(steps)
    executor = PipelineExecutor(graph, state_dir)
    LOGGER.info("[pipeline] running '%s'", pipeline_name)
    executor.run(_execute_step)


@pipeline_app.command()
def resume(pipeline_file: Path = typer.Argument(..., exists=True)) -> None:
    """Resume a previously started pipeline."""

    run(pipeline_file)


@pipeline_app.command()
def viz(pipeline_file: Path = typer.Argument(..., exists=True)) -> None:  # pragma: no cover - optional feature
    """Print a simple textual representation of the pipeline graph."""

    payload = _load_yaml(pipeline_file)
    pipeline_payload = payload.get("pipeline")
    name, steps = _load_pipeline_steps(pipeline_payload)
    typer.echo(f"Pipeline: {name}")
    for step in steps:
        deps = ", ".join(step.after) if step.after else "<root>"
        typer.echo(f" - {step.step_id} ({step.uses}) after [{deps}]")


@models_app.command("register")
def models_register(name: str, uri: str, meta: Path) -> None:
    """Register a model artifact with the SQLite registry."""

    meta_payload = json.loads(meta.read_text())
    version = registry_register(name, uri, meta_payload, db_path=DEFAULT_DB_PATH)
    typer.echo(f"registered {name} version {version}")


@models_app.command("resolve")
def models_resolve(reference: str) -> None:
    """Resolve a model reference to a concrete URI."""

    uri = registry_resolve(reference, db_path=DEFAULT_DB_PATH)
    typer.echo(uri)


@models_app.command("promote")
def models_promote(name: str, version: str, tag: str = typer.Option("latest")) -> None:
    """Promote a registered model version to a tag."""

    registry_promote(name, version, tag, db_path=DEFAULT_DB_PATH)
    typer.echo(f"promoted {name}:{tag} -> {version}")


@serve_app.command("start")
def serve_start(
    model: Path = typer.Argument(..., exists=True),
    backend: str = typer.Option("vllm"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    config: Optional[Path] = typer.Option(None),
    tensor_parallel: Optional[int] = typer.Option(None),
    dry_run: bool = typer.Option(False),
) -> None:
    """Start a serving backend."""

    spec = ServeRunSpec(
        backend=backend,
        model_path=model,
        host=host,
        port=port,
        tensor_parallel=tensor_parallel,
        config_path=config,
        dry_run=dry_run,
    )
    handle = RUNTIME.serve_runner.start(spec)
    ACTIVE_SERVE_HANDLES[port] = handle
    typer.echo(f"Serve backend '{backend}' started on {handle.handle.endpoint}")  # type: ignore[attr-defined]

    if handle.handle.process is not None and not dry_run:
        typer.echo("Press Ctrl+C to stop server")
        try:
            handle.handle.process.wait()
        except KeyboardInterrupt:
            typer.echo("Stopping server...")
            RUNTIME.serve_runner.stop(handle)


@serve_app.command("stop")
def serve_stop(port: int) -> None:
    """Stop a running serve backend started from this CLI session."""

    handle = ACTIVE_SERVE_HANDLES.pop(port, None)
    if not handle:
        typer.echo(f"No active serve handle tracked on port {port}")
        raise typer.Exit(code=1)
    RUNTIME.serve_runner.stop(handle)
    typer.echo(f"Stopped serve backend on port {port}")


@serve_app.command("health")
def serve_health(port: int) -> None:
    """Check the health of a running serve backend."""

    handle = ACTIVE_SERVE_HANDLES.get(port)
    if not handle:
        typer.echo(f"No active serve handle tracked on port {port}")
        raise typer.Exit(code=1)
    healthy = RUNTIME.serve_runner.health_check(handle)
    typer.echo("healthy" if healthy else "unhealthy")


@app.command()
def setup_env() -> None:  # pragma: no cover - depends on user environment
    """Placeholder for environment setup checks."""

    typer.echo("setup-env is currently a placeholder. Configure environments manually.")


if __name__ == "__main__":  # pragma: no cover
    app()
