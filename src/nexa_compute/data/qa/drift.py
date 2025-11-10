"""Distribution drift detection stubs (KL/JS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = [
    "DriftReport",
    "compute_drift",
]


@dataclass(frozen=True)
class DriftReport:
    """Placeholder report for drift analysis."""

    metric: str
    score: float
    threshold: float
    flagged: bool
    details: Mapping[str, float]


def compute_drift(
    baseline_distribution: Mapping[str, float],
    candidate_distribution: Mapping[str, float],
    *,
    metric: str = "kl",
    threshold: float = 0.5,
) -> DriftReport:
    """Return a stub drift report with zero score.

    The function preserves the API expected by downstream components while the
    full statistical implementation is built. Callers can inspect ``flagged``
    to decide whether remediation is needed.
    """

    _ = (baseline_distribution, candidate_distribution)  # Placeholder for future logic.
    return DriftReport(
        metric=metric,
        score=0.0,
        threshold=threshold,
        flagged=False,
        details={},
    )
