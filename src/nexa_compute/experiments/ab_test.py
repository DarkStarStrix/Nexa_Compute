"""Framework for running A/B tests on models."""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexa_compute.monitoring.alerts import AlertSeverity, send_alert

LOGGER = logging.getLogger(__name__)


@dataclass
class Experiment:
    name: str
    variants: List[str] # List of model versions
    weights: List[float] # Traffic split weights
    metrics: List[str] # Metrics to track (e.g., "ctr", "latency")
    start_time: float
    end_time: Optional[float] = None
    active: bool = True
    description: str = ""


class ABTestManager:
    """Manages active A/B tests and variant assignment."""

    def __init__(self) -> None:
        self._experiments: Dict[str, Experiment] = {}
        self._results: Dict[str, Dict[str, Dict[str, float]]] = {} # exp -> variant -> metric -> sum

    def create_experiment(
        self,
        name: str,
        variants: List[str],
        weights: Optional[List[float]] = None,
        metrics: Optional[List[str]] = None,
        description: str = "",
    ) -> Experiment:
        """Start a new A/B test."""
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")
            
        if weights is None:
            weights = [1.0 / len(variants)] * len(variants)
            
        if len(weights) != len(variants):
            raise ValueError("Weights must match number of variants")
            
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        import time
        exp = Experiment(
            name=name,
            variants=variants,
            weights=weights,
            metrics=metrics or ["latency"],
            start_time=time.time(),
            description=description,
        )
        self._experiments[name] = exp
        self._results[name] = {v: {m: 0.0 for m in exp.metrics} for v in variants}
        self._results[name]["counts"] = {v: 0 for v in variants} # type: ignore
        
        send_alert(
            title="A/B Test Started",
            message=f"Experiment '{name}' started with variants {variants}",
            severity=AlertSeverity.INFO,
            source="ab_testing",
        )
        return exp

    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Deterministically assign a variant to a user."""
        exp = self._experiments.get(experiment_name)
        if not exp or not exp.active:
            return "default" # Or raise error
            
        # Deterministic hash for stable assignment
        hash_input = f"{experiment_name}:{user_id}"
        hash_val = int(hashlib.sha256(hash_input.encode("utf-8")).hexdigest(), 16)
        rand_val = (hash_val % 10000) / 10000.0
        
        cumulative = 0.0
        for i, weight in enumerate(exp.weights):
            cumulative += weight
            if rand_val <= cumulative:
                return exp.variants[i]
                
        return exp.variants[-1] # Fallback

    def log_metric(self, experiment_name: str, variant: str, metric: str, value: float) -> None:
        """Record a metric for an experiment variant."""
        if experiment_name not in self._results:
            return
            
        if variant not in self._results[experiment_name]:
            return
            
        if metric in self._results[experiment_name][variant]:
            self._results[experiment_name][variant][metric] += value
            
        if metric == "latency": # Implicitly count requests
             self._results[experiment_name]["counts"][variant] += 1 # type: ignore

    def get_results(self, experiment_name: str) -> Dict[str, Any]:
        """Get current results for an experiment."""
        if experiment_name not in self._results:
            return {}
            
        raw = self._results[experiment_name]
        counts = raw["counts"]
        
        summary = {}
        for variant in raw:
            if variant == "counts":
                continue
            
            count = counts[variant] # type: ignore
            if count == 0:
                summary[variant] = {m: 0.0 for m in raw[variant]}
            else:
                summary[variant] = {m: val / count for m, val in raw[variant].items()}
                
        return summary

# Global instance
_AB_MANAGER = ABTestManager()

def get_ab_manager() -> ABTestManager:
    return _AB_MANAGER

