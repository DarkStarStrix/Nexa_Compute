"""Evaluation orchestration routines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.config import load_config  # type: ignore
from nexa_compute.evaluation import Evaluator  # type: ignore
from nexa_compute.data import DataPipeline  # type: ignore
from nexa_compute.models import DEFAULT_MODEL_REGISTRY  # type: ignore
from nexa_compute.utils.checkpoint import load_checkpoint  # type: ignore


def evaluate_checkpoint(config_path: Path, checkpoint: Optional[Path] = None) -> Path:
    cfg = load_config(config_path)
    pipeline = DataPipeline(cfg.data)
    dataloader = pipeline.dataloader("validation", batch_size=cfg.evaluation.batch_size)
    model = DEFAULT_MODEL_REGISTRY.build(cfg.model)
    if checkpoint:
        state = load_checkpoint(checkpoint)
        model.load_state_dict(state["model_state"])
    evaluator = Evaluator(cfg.evaluation)
    metrics = evaluator.evaluate(model, dataloader)
    run_dir = cfg.output_directory()
    output = run_dir / "eval_report.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[nexa-eval] Metrics saved to {output}")
    return output
