"""Sync datasets from S3 using the AWS CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Sync data from S3 buckets into the local data directory")


@app.command()
def sync(
    bucket: str = typer.Argument(..., help="Name of the S3 bucket"),
    prefix: str = typer.Option("", help="Optional prefix inside the bucket"),
    destination: Path = typer.Option(Path("data/raw"), help="Local destination directory"),
    profile: Optional[str] = typer.Option(None, help="AWS CLI profile to use"),
    dry_run: bool = typer.Option(False, help="Show actions without performing download"),
) -> None:
    """Mirror objects from S3 to the local filesystem using the AWS CLI."""

    destination.mkdir(parents=True, exist_ok=True)
    s3_uri = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
    cmd = ["aws", "s3", "sync", s3_uri, str(destination)]
    if profile:
        cmd.extend(["--profile", profile])
    if dry_run:
        cmd.append("--dryrun")
    typer.echo(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise typer.Exit(code=1) from exc
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Sync failed with exit code {exc.returncode}")
        raise typer.Exit(code=exc.returncode)


if __name__ == "__main__":
    app()
