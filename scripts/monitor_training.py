"""Monitor active runs by tailing structured logs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def tail(
    log_dir: Path = typer.Option(Path("logs"), help="Directory with training logs"),
    refresh_seconds: float = typer.Option(2.0, help="Polling interval"),
    key: Optional[str] = typer.Option(None, help="Filter by metric key"),
) -> None:
    log_path = log_dir / "train.log"
    if not log_path.exists():
        typer.echo(f"Log file not found: {log_path}")
        raise typer.Exit(code=1)
    typer.echo(f"Tailing {log_path} (Ctrl+C to stop)")
    with log_path.open("r", encoding="utf-8") as handle:
        # Seek to end
        handle.seek(0, 2)
        try:
            while True:
                line = handle.readline()
                if not line:
                    time.sleep(refresh_seconds)
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    typer.echo(line.strip())
                    continue
                if key and key not in payload:
                    continue
                typer.echo(json.dumps(payload, indent=2))
        except KeyboardInterrupt:
            typer.echo("Stopped monitoring")


if __name__ == "__main__":
    app()
