"""Statistical utilities for detecting distribution drift."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)


class DriftDetector:
    """Detects drift between a reference distribution and current data."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def detect_drift(
        self,
        reference: Union[List[float], np.ndarray],
        current: Union[List[float], np.ndarray],
        method: str = "ks_2samp",
    ) -> Dict[str, Any]:
        """Detect drift using statistical tests.

        Args:
            reference: Baseline data distribution (e.g., training data).
            current: New data distribution (e.g., inference data).
            method: Statistical test to use ("ks_2samp", "chisquare", "psi").

        Returns:
            Dictionary with drift detected status, p-value, and distance.
        """
        if len(reference) == 0 or len(current) == 0:
            return {"drift_detected": False, "reason": "insufficient_data"}

        ref_arr = np.array(reference)
        curr_arr = np.array(current)

        if method == "ks_2samp":
            # Kolmogorov-Smirnov test for continuous distributions
            statistic, p_value = stats.ks_2samp(ref_arr, curr_arr)
        elif method == "psi":
            # Population Stability Index
            statistic = self._calculate_psi(ref_arr, curr_arr)
            p_value = 1.0 # PSI doesn't have a p-value
            # Typically PSI > 0.2 indicates significant drift
            is_drift = statistic > 0.2
            return {
                "drift_detected": is_drift,
                "method": "psi",
                "statistic": float(statistic),
                "p_value": None,
            }
        else:
            raise ValueError(f"Unknown drift detection method: {method}")

        is_drift = p_value < self.threshold
        return {
            "drift_detected": is_drift,
            "method": method,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "threshold": self.threshold,
        }

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        cnt_expected = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0]
        # Use the same bins for actual
        cnt_actual = np.histogram(actual, bins=np.percentile(expected, breakpoints))[0]

        # Avoid division by zero
        cnt_expected = np.where(cnt_expected == 0, 0.0001, cnt_expected)
        cnt_actual = np.where(cnt_actual == 0, 0.0001, cnt_actual)

        per_expected = cnt_expected / len(expected)
        per_actual = cnt_actual / len(actual)

        psi_values = (per_actual - per_expected) * np.log(per_actual / per_expected)
        return np.sum(psi_values)

