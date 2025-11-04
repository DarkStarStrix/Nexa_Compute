"""Very light-weight weakness analysis based on evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def identify_weaknesses(config_path: Path) -> Dict[str, float]:
    """Identify weaknesses from evaluation reports."""
    from pathlib import Path
    import json
    
    # Look for evaluation reports in organized structure
    eval_dir = Path("data/processed/evaluation/reports")
    if not eval_dir.exists():
        return {"message": "run evaluation before feedback - no evaluation reports found"}
    
    # Find most recent evaluation report
    reports = sorted(eval_dir.glob("eval_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        return {"message": "run evaluation before feedback - no evaluation reports found"}
    
    report = reports[0]
    metrics = json.loads(report.read_text(encoding="utf-8"))
    weaknesses = {k: v for k, v in metrics.items() if k.endswith("loss") or (isinstance(v, (int, float)) and v < 0.8)}
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
