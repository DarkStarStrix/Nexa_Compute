"""Generate model predictions for evaluation tasks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.config import load_config  # type: ignore
from nexa_compute.data import DataPipeline  # type: ignore
from nexa_compute.models import DEFAULT_MODEL_REGISTRY  # type: ignore
from nexa_compute.utils.checkpoint import load_checkpoint  # type: ignore


def generate_predictions(config_path: Path, checkpoint: Optional[Path] = None) -> Tuple[List[List[float]], List[int]]:
    cfg = load_config(config_path)
    pipeline = DataPipeline(cfg.data)
    dataloader = pipeline.dataloader("validation", batch_size=cfg.evaluation.batch_size)
    model = DEFAULT_MODEL_REGISTRY.build(cfg.model)
    if checkpoint:
        state = load_checkpoint(checkpoint)
        model.load_state_dict(state["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    probs: List[List[float]] = []
    labels: List[int] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_probs = torch.softmax(outputs, dim=1).cpu().tolist()
            probs.extend(batch_probs)
            labels.extend(targets.tolist())
    return probs, labels
