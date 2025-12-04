"""Python wrapper for nexa_stats Rust extension."""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    import nexa_stats as _rust
except ImportError:
    _rust = None


class StatsCoreError(Exception):
    """Base exception for stats core errors."""
    pass


class NexaStatsCore:
    """Interface to the high-performance Rust statistics engine."""

    @property
    def available(self) -> bool:
        return _rust is not None

    def _check_available(self) -> None:
        if not self.available:
            raise StatsCoreError("nexa_stats Rust extension not installed or failed to load.")

    def ks_test(self, ref_data: List[float], cur_data: List[float]) -> float:
        """Compute Kolmogorov-Smirnov test statistic."""
        self._check_available()
        return _rust.ks_test(ref_data, cur_data)

    def psi(self, ref_data: List[float], cur_data: List[float]) -> float:
        """Compute Population Stability Index."""
        self._check_available()
        return _rust.psi(ref_data, cur_data)

    def chi_square(self, observed: List[float], expected: List[float]) -> float:
        """Compute Chi-Square test statistic."""
        self._check_available()
        return _rust.chi_square(observed, expected)

    def histogram(self, data: List[float], bins: int = 10) -> Dict[str, List[Any]]:
        """Compute histogram."""
        self._check_available()
        try:
            json_str = _rust.compute_histogram(data, bins)
            return json.loads(json_str)
        except Exception as e:
            raise StatsCoreError(f"Histogram failed: {e}") from e

    def reduce(self, data: List[float]) -> Dict[str, float]:
        """Compute basic reduction metrics."""
        self._check_available()
        try:
            json_str = _rust.compute_reductions(data)
            return json.loads(json_str)
        except Exception as e:
            raise StatsCoreError(f"Reduction failed: {e}") from e


# Global instance
stats_core = NexaStatsCore()
