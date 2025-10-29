"""Create distilled datasets from teacher predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


def materialize_distilled_dataset(probabilities: Iterable[torch.Tensor], targets: Iterable[torch.Tensor], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    distilled = []
    for probs, target in zip(probabilities, targets):
        distilled.append({
            "teacher_probs": probs.tolist(),
            "target": int(target.item()),
        })
    path = output_dir / "distilled_dataset.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(distilled, handle, indent=2)
    return path
