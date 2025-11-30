"""CLI for inference server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .batch_processor import process_directory
from .server import serve_model
from .spectral_server import serve_spectral_model

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


@app.command()
def serve_spectral(
    checkpoint: Path = typer.Option(..., help="Path to spectral model checkpoint"),
    config: Optional[Path] = typer.Option(None, help="Path to model config YAML"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL"),
    qdrant_api_key: Optional[str] = typer.Option(None, help="Qdrant API key"),
    collection_name: str = typer.Option("nexa_spectra", help="Qdrant collection name"),
    embedding_dim: int = typer.Option(768, help="Embedding dimension"),
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
) -> None:
    """Serve spectral inference model with embedding and search endpoints."""
    serve_spectral_model(
        checkpoint,
        config,
        qdrant_url,
        qdrant_api_key,
        collection_name,
        embedding_dim,
        host,
        port,
    )


@app.command()
def process_batch(
    input_dir: Path = typer.Option(..., help="Directory containing shard files"),
    output_dir: Path = typer.Option(..., help="Directory to save embeddings"),
    model_path: Path = typer.Option(..., help="Path to model checkpoint"),
    collection_name: Optional[str] = typer.Option(None, help="Qdrant collection name"),
    qdrant_url: Optional[str] = typer.Option(None, help="Qdrant server URL"),
    device: str = typer.Option("cuda", help="Device to run inference on"),
    batch_size: int = typer.Option(32, help="Batch size for processing"),
) -> None:
    """Process batch of shards for embedding generation."""
    stats = process_directory(
        input_dir,
        output_dir,
        model_path,
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        device=device,
        batch_size=batch_size,
    )
    typer.echo(f"Processed {stats['shards_processed']} shards with {stats['total_spectra']} total spectra")


if __name__ == "__main__":
    app()

