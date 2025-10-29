"""Serve a minimalist leaderboard of runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Nexa Compute Leaderboard")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    rows = _load_runs()
    html_rows = "".join(
        f"<tr><td>{row['run']}</td><td>{row['metric']}</td><td>{row['value']:.4f}</td></tr>" for row in rows
    )
    return f"""
    <html>
      <head><title>Nexa Leaderboard</title></head>
      <body>
        <h1>Nexa Leaderboard</h1>
        <table border='1' cellpadding='6'>
          <tr><th>Run</th><th>Metric</th><th>Value</th></tr>
          {html_rows}
        </table>
      </body>
    </html>
    """


def _load_runs() -> List[Dict[str, float]]:
    runs_dir = Path("runs/manifests")
    rows: List[Dict[str, float]] = []
    for path in runs_dir.glob("run_manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for metric, value in payload.get("metrics", {}).items():
            rows.append({"run": payload.get("run_dir", path.parent.name), "metric": metric, "value": value})
    return rows


def serve_leaderboard(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port)
