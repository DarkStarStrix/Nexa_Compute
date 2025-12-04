"""Evaluation utilities and metric helpers.

Responsibility: Executes model evaluation workflows including metric computation, prediction saving, 
and generation of evaluation reports and visualizations.
"""

from .metrics import MetricRegistry, compute_metrics
from .evaluator import Evaluator

__all__ = ["MetricRegistry", "compute_metrics", "Evaluator"]
