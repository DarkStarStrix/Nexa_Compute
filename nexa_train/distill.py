"""Knowledge distillation helpers connecting training and data modules."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from nexa_data.distill_materialize import materialize_distilled_dataset
from nexa_eval.generate import generate_predictions


def distill_teacher(config_path: Path, checkpoint: Optional[Path], output_dir: Path) -> Path:
    probabilities, targets = generate_predictions(config_path, checkpoint)
    stacked_probs = [torch.tensor(prob) for prob in probabilities]
    stacked_targets = [torch.tensor(target) for target in targets]
    return materialize_distilled_dataset(stacked_probs, stacked_targets, output_dir)
