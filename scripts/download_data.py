"""Utility to download datasets from HTTP locations."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def download(
    url: str = typer.Argument(..., help="Source URL"),
    output_dir: Path = typer.Option(Path("data/raw"), help="Download directory"),
    filename: Optional[str] = typer.Option(None, help="Optional file name"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_name = filename or Path(url).name
    target_path = output_dir / target_name
    typer.echo(f"Downloading {url} → {target_path}")
    urllib.request.urlretrieve(url, target_path)
    checksum = _sha256(target_path)
    typer.echo(f"Saved {target_path} (sha256: {checksum})")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    app()
