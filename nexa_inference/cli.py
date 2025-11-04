"""CLI for inference server."""

from __future__ import annotations

from pathlib import Path

import typer

from .server import serve_model

app = typer.Typer()


@app.command()
def serve(
    checkpoint: Path = typer.Argument(..., exists=True, help="Path to model checkpoint"),
    config: Optional[Path] = typer.Option(None, help="Path to model config YAML"),
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
) -> None:
    """Serve model inference via FastAPI."""
    serve_model(checkpoint, config, host=host, port=port)


if __name__ == "__main__":
    app()

