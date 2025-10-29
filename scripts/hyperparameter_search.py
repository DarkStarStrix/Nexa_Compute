"""Simple random search tuner for Nexa Compute."""

from __future__ import annotations

import json
import math
import random
from typing import Dict, List

from nexa_compute.config.schema import TrainingConfig
from nexa_compute.orchestration import TrainingPipeline


def _sample_learning_rate(base_lr: float) -> float:
    log_lr = math.log10(base_lr)
    return 10 ** random.uniform(log_lr - 1, log_lr + 1)


def _sample_dropout(base: float) -> float:
    return float(min(0.7, max(0.0, random.gauss(base, 0.1))))


def random_search(config: TrainingConfig, trials: int = 5) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for idx in range(trials):
        trial_cfg = config.model_copy(deep=True)
        lr = _sample_learning_rate(trial_cfg.training.optimizer.lr)
        dropout = _sample_dropout(trial_cfg.model.parameters.get("dropout", 0.2))
        trial_cfg.training.optimizer.lr = lr
        trial_cfg.model.parameters["dropout"] = dropout
        trial_cfg.training.epochs = min(trial_cfg.training.epochs, 5)
        trial_cfg.experiment.name = f"{config.experiment.name}_tune_{idx}"

        pipeline = TrainingPipeline(trial_cfg)
        artifacts = pipeline.run(enable_evaluation=True)
        results.append(
            {
                "trial": idx,
                "lr": lr,
                "dropout": dropout,
                "metrics": artifacts.metrics,
                "run_dir": str(artifacts.run_dir),
            }
        )
    return results
