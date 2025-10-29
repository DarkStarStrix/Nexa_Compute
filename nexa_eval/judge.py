"""Simple rubric-based judging for evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Rubric:
    metric: str
    threshold: float
    direction: str = "max"


def judge_metrics(metrics: Dict[str, float], rubrics: List[Rubric]) -> Dict[str, bool]:
    verdicts = {}
    for rubric in rubrics:
        value = metrics.get(rubric.metric)
        if value is None:
            verdicts[rubric.metric] = False
            continue
        if rubric.direction == "max":
            verdicts[rubric.metric] = value >= rubric.threshold
        else:
            verdicts[rubric.metric] = value <= rubric.threshold
    return verdicts
