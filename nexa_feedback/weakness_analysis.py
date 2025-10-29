"""Very light-weight weakness analysis based on evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def identify_weaknesses(config_path: Path) -> Dict[str, float]:
    run_dir = _config_to_run_dir(config_path)
    report = run_dir / "eval_report.json"
    if not report.exists():
        return {"message": "run evaluation before feedback"}
    metrics = json.loads(report.read_text(encoding="utf-8"))
    weaknesses = {k: v for k, v in metrics.items() if k.endswith("loss") or v < 0.8}
    return weaknesses


def _config_to_run_dir(config_path: Path) -> Path:
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from nexa_compute.config import load_config  # type: ignore

    cfg = load_config(config_path)
    return cfg.output_directory()
