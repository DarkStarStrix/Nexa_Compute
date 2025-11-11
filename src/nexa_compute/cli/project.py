"""CLI utilities for managing NexaCompute projects."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import typer

from nexa_compute.core.project_registry import (
    DEFAULT_PROJECT_REGISTRY,
    ProjectRegistryError,
    validate_project_slug,
)

app = typer.Typer(no_args_is_help=True, help="Project scaffolding utilities.")

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECTS_DIR = REPO_ROOT / "projects"
TEMPLATE_DIR = PROJECTS_DIR / "_template"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
LOGS_DIR = REPO_ROOT / "logs"


def _default_display_name(slug: str) -> str:
    return slug.replace("_", " ").title()


def _default_run_prefix(slug: str) -> str:
    parts = slug.split("_")
    prefix = "".join(part[:3] for part in parts if part)
    return prefix or slug[:3]


def _render_placeholders(root: Path, replacements: Dict[str, str]) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            payload = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary file
        original = payload
        for key, value in replacements.items():
            payload = payload.replace(f"{{{{{key}}}}}", value)
        if payload != original:
            path.write_text(payload, encoding="utf-8")


def _create_support_directories(project_slug: str) -> None:
    (DATA_DIR / "raw" / project_slug).mkdir(parents=True, exist_ok=True)
    processed_root = DATA_DIR / "processed" / project_slug
    for sub in ["distillation", "tool_protocol", "training", "evaluation"]:
        (processed_root / sub).mkdir(parents=True, exist_ok=True)
    artifact_root = ARTIFACTS_DIR / project_slug
    for sub in ["checkpoints", "eval", "runs"]:
        (artifact_root / sub).mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / project_slug).mkdir(parents=True, exist_ok=True)


@app.command()
def scaffold(
    project_slug: str = typer.Argument(..., help="Project slug (lowercase, alphanumeric, underscores)."),
    display_name: str = typer.Option(None, "--name", help="Human friendly project name."),
    owner: str = typer.Option("unknown", "--owner", help="Primary owner or team."),
    description: str = typer.Option("", "--description", help="Short description of the project."),
    base_model: str = typer.Option("tiiuae/Falcon3-10B-Base", "--base-model", help="Default base model identifier."),
    run_prefix: str = typer.Option(None, "--run-prefix", help="Prefix for run identifiers."),
) -> None:
    """Create a new project skeleton from the template."""

    validate_project_slug(project_slug)
    if any(meta.slug == project_slug for meta in DEFAULT_PROJECT_REGISTRY.list()):
        raise typer.BadParameter(f"Project '{project_slug}' already exists.")

    target_dir = PROJECTS_DIR / project_slug
    if target_dir.exists():
        raise typer.BadParameter(f"Directory {target_dir} already exists.")

    if not TEMPLATE_DIR.exists():
        raise typer.BadParameter(f"Template directory missing: {TEMPLATE_DIR}")

    display = display_name or _default_display_name(project_slug)
    prefix = run_prefix or _default_run_prefix(project_slug)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    typer.echo(f"Scaffolding project '{project_slug}' at {target_dir}...")
    shutil.copytree(TEMPLATE_DIR, target_dir)

    replacements = {
        "project_slug": project_slug,
        "project_name": display,
        "project_owner": owner,
        "project_description": description or f"{display} project",
        "base_model": base_model,
        "timestamp": timestamp,
        "run_prefix": prefix,
    }
    _render_placeholders(target_dir, replacements)

    _create_support_directories(project_slug)

    DEFAULT_PROJECT_REGISTRY.refresh()
    typer.echo(f"✓ Project '{project_slug}' created.")
    typer.echo("Next steps:")
    typer.echo(f"  1. Review manifest at {target_dir / 'manifests/project_manifest.json'}")
    typer.echo(f"  2. Populate data in {DATA_DIR / 'raw' / project_slug}")
    typer.echo(f"  3. Add training configs to {target_dir / 'configs'}")
    typer.echo("  4. Update docs and pipelines as needed.")


def _validate_project(meta) -> List[str]:
    errors: List[str] = []
    manifest_path = meta.manifests_dir / "project_manifest.json"
    if not manifest_path.exists():
        errors.append(f"[{meta.slug}] Missing manifest file: {manifest_path}")
    required_dirs = [
        ("configs", meta.configs_dir),
        ("docs", meta.docs_dir),
        ("pipelines", meta.pipelines_dir),
        ("raw data", meta.raw_data_dir),
        ("processed data", meta.processed_data_dir),
        ("artifacts", meta.artifacts_dir),
    ]
    for label, path in required_dirs:
        if not path.exists():
            errors.append(f"[{meta.slug}] Missing {label} directory: {path}")
    return errors


@app.command()
def validate(project_slug: Optional[str] = typer.Option(None, "--project-slug", help="Validate a single project.")) -> None:
    """Validate project manifests and directory structure."""

    DEFAULT_PROJECT_REGISTRY.refresh()
    slugs = [project_slug] if project_slug else DEFAULT_PROJECT_REGISTRY.slugs()
    if not slugs:
        typer.echo("No projects registered.")
        raise typer.Exit(0)

    errors: List[str] = []
    for slug in slugs:
        try:
            meta = DEFAULT_PROJECT_REGISTRY.get(slug)
        except ProjectRegistryError as exc:
            errors.append(str(exc))
            continue
        errors.extend(_validate_project(meta))

    if errors:
        for message in errors:
            typer.echo(f"❌ {message}")
        raise typer.Exit(code=1)

    typer.echo("✓ All projects validated successfully.")


if __name__ == "__main__":
    app()

