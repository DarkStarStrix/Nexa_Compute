"""Semantic consistency checker for dataset drift detection."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SemanticDriftThresholds:
    """Thresholds for semantic drift detection."""

    peak_count_ks_pvalue: float = 0.05
    precursor_mz_ks_pvalue: float = 0.05
    intensity_quantile_max_diff: float = 0.1
    adduct_chi2_pvalue: float = 0.05
    charge_chi2_pvalue: float = 0.05


@dataclass
class DistributionStats:
    """Statistics for a distribution."""

    values: List[float] = field(default_factory=list)
    histogram: Optional[Dict[str, int]] = None
    quantiles: Optional[List[float]] = None

    def update(self, value: float) -> None:
        """Add a value to the distribution."""
        self.values.append(value)

    def compute_quantiles(self, percentiles: List[float] = None) -> List[float]:
        """Compute quantiles of the distribution."""
        if not self.values:
            return []
        if percentiles is None:
            percentiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        arr = np.array(self.values)
        return np.percentile(arr, [p * 100 for p in percentiles]).tolist()

    def compute_histogram(self, bins: int = 50) -> Dict[str, int]:
        """Compute histogram of the distribution."""
        if not self.values:
            return {}
        arr = np.array(self.values)
        hist, bin_edges = np.histogram(arr, bins=bins)
        return {
            f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}": int(count)
            for i, count in enumerate(hist)
        }


@dataclass
class BaselineDistributions:
    """Baseline distributions for drift detection."""

    peak_counts: List[int] = field(default_factory=list)
    precursor_mzs: List[float] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    adducts: Dict[str, int] = field(default_factory=dict)
    charges: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "peak_counts": self.peak_counts,
            "precursor_mzs": self.precursor_mzs,
            "intensities": self.intensities,
            "adducts": self.adducts,
            "charges": {str(k): v for k, v in self.charges.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BaselineDistributions":
        """Create from dictionary."""
        return cls(
            peak_counts=data.get("peak_counts", []),
            precursor_mzs=data.get("precursor_mzs", []),
            intensities=data.get("intensities", []),
            adducts=data.get("adducts", {}),
            charges={int(k): v for k, v in data.get("charges", {}).items()},
        )


class SemanticChecker:
    """Semantic consistency checker with drift detection."""

    def __init__(
        self,
        thresholds: Optional[SemanticDriftThresholds] = None,
        baseline: Optional[BaselineDistributions] = None,
        window_size: int = 10000,
    ):
        """Initialize semantic checker.

        Args:
            thresholds: Drift detection thresholds
            baseline: Baseline distributions for comparison
            window_size: Rolling window size for drift detection
        """
        self.thresholds = thresholds or SemanticDriftThresholds()
        self.baseline = baseline
        self.window_size = window_size

        self.rolling_peak_counts: List[int] = []
        self.rolling_precursor_mzs: List[float] = []
        self.rolling_intensities: List[float] = []
        self.rolling_adducts: Dict[str, int] = defaultdict(int)
        self.rolling_charges: Dict[int, int] = defaultdict(int)

        self.all_peak_counts: List[int] = []
        self.all_precursor_mzs: List[float] = []
        self.all_intensities: List[float] = []
        self.all_adducts: Dict[str, int] = defaultdict(int)
        self.all_charges: Dict[int, int] = defaultdict(int)

        self.drift_detected = False
        self.drift_reasons: List[str] = []

    def add_sample(self, record: Dict) -> None:
        """Add a sample to the checker.

        Args:
            record: Sample record dictionary
        """
        peak_count = len(record.get("mzs", []))
        precursor_mz = record.get("precursor_mz", 0.0)
        intensities = record.get("ints", [])
        adduct = record.get("adduct", "unknown")
        charge = record.get("charge", 0)

        self.rolling_peak_counts.append(peak_count)
        self.rolling_precursor_mzs.append(precursor_mz)
        self.rolling_intensities.extend(intensities)
        self.rolling_adducts[adduct] += 1
        self.rolling_charges[charge] += 1

        self.all_peak_counts.append(peak_count)
        self.all_precursor_mzs.append(precursor_mz)
        self.all_intensities.extend(intensities)
        self.all_adducts[adduct] += 1
        self.all_charges[charge] += 1

        if len(self.rolling_peak_counts) >= self.window_size:
            self._check_drift()
            self._reset_rolling_window()

    def _reset_rolling_window(self) -> None:
        """Reset rolling window statistics."""
        self.rolling_peak_counts.clear()
        self.rolling_precursor_mzs.clear()
        self.rolling_intensities.clear()
        self.rolling_adducts.clear()
        self.rolling_charges.clear()

    def _check_drift(self) -> None:
        """Check for semantic drift in rolling window."""
        if not self.baseline:
            return

        reasons = []

        if self.baseline.peak_counts:
            ks_stat, pvalue = stats.ks_2samp(
                self.baseline.peak_counts,
                self.rolling_peak_counts,
            )
            if pvalue < self.thresholds.peak_count_ks_pvalue:
                reasons.append(
                    f"Peak count distribution drift (KS p={pvalue:.4f})"
                )

        if self.baseline.precursor_mzs:
            ks_stat, pvalue = stats.ks_2samp(
                self.baseline.precursor_mzs,
                self.rolling_precursor_mzs,
            )
            if pvalue < self.thresholds.precursor_mz_ks_pvalue:
                reasons.append(
                    f"Precursor m/z distribution drift (KS p={pvalue:.4f})"
                )

        if self.baseline.intensities:
            baseline_quantiles = np.percentile(
                np.array(self.baseline.intensities),
                [25, 50, 75],
            )
            rolling_quantiles = np.percentile(
                np.array(self.rolling_intensities),
                [25, 50, 75],
            )
            max_diff = np.abs(baseline_quantiles - rolling_quantiles).max()
            if max_diff > self.thresholds.intensity_quantile_max_diff:
                reasons.append(
                    f"Intensity quantile drift (max diff={max_diff:.4f})"
                )

        if self.baseline.adducts:
            if self._check_categorical_drift(
                self.baseline.adducts,
                self.rolling_adducts,
                self.thresholds.adduct_chi2_pvalue,
            ):
                reasons.append("Adduct distribution drift")

        if self.baseline.charges:
            if self._check_categorical_drift(
                self.baseline.charges,
                self.rolling_charges,
                self.thresholds.charge_chi2_pvalue,
            ):
                reasons.append("Charge distribution drift")

        if reasons:
            self.drift_detected = True
            self.drift_reasons.extend(reasons)
            logger.warning(f"Semantic drift detected: {', '.join(reasons)}")

    def _check_categorical_drift(
        self,
        baseline: Dict,
        current: Dict,
        pvalue_threshold: float,
    ) -> bool:
        """Check for drift in categorical distribution using chi-square test."""
        all_keys = set(baseline.keys()) | set(current.keys())
        if not all_keys:
            return False

        baseline_counts = [baseline.get(k, 0) for k in all_keys]
        current_counts = [current.get(k, 0) for k in all_keys]

        total_baseline = sum(baseline_counts)
        total_current = sum(current_counts)

        if total_baseline == 0 or total_current == 0:
            return False

        expected = [
            (baseline_counts[i] / total_baseline) * total_current
            for i in range(len(all_keys))
        ]

        if any(e == 0 for e in expected):
            return False

        chi2, pvalue = stats.chisquare(current_counts, expected)

        return pvalue < pvalue_threshold

    def check_final_drift(self) -> bool:
        """Check for drift using all collected data."""
        if not self.baseline:
            return False

        self._check_drift()
        return self.drift_detected

    def get_distributions(self) -> BaselineDistributions:
        """Get current distributions."""
        return BaselineDistributions(
            peak_counts=self.all_peak_counts.copy(),
            precursor_mzs=self.all_precursor_mzs.copy(),
            intensities=self.all_intensities.copy(),
            adducts=dict(self.all_adducts),
            charges=dict(self.all_charges),
        )

    def get_drift_status(self) -> Dict:
        """Get drift detection status."""
        return {
            "drift_detected": self.drift_detected,
            "drift_reasons": self.drift_reasons.copy(),
            "sample_count": len(self.all_peak_counts),
        }

