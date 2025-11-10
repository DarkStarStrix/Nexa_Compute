"""CLI for inference server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .server import serve_model

app = typer.Typer()


@app.command()
def serve(
    checkpoint: Optional[Path] = typer.Option(None, help="Path to model checkpoint or artifact"),
    config: Optional[Path] = typer.Option(None, help="Path to model config YAML"),
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reference: Optional[str] = typer.Option(None, help="Registry reference (name[:tag])"),
) -> None:
    """Serve model inference via FastAPI."""
    if checkpoint is None and reference is None:
        raise typer.BadParameter("Provide either --checkpoint or --reference")
    serve_model(checkpoint, config, reference=reference, host=host, port=port)


if __name__ == "__main__":
    app()

