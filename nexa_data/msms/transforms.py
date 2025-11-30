"""Canonical transform and cleaning functions."""

import numpy as np
from typing import Dict, Optional

from .config import PreprocessingConfig
from .metrics import PipelineMetrics

# Try to import Rust validator
try:
    import msms_rust
    RUST_VALIDATOR_AVAILABLE = True
except ImportError:
    RUST_VALIDATOR_AVAILABLE = False


def clean_and_canonicalize(
    sample_id: str,
    raw: dict,
    cfg: PreprocessingConfig,
    metrics: Optional[PipelineMetrics] = None,
) -> Optional[dict]:
    """Clean and canonicalize a spectrum record.

    Args:
        sample_id: Sample identifier
        raw: Raw spectrum record from HDF5
        cfg: Preprocessing configuration
        metrics: Optional metrics tracker

    Returns:
        Canonical record dict or None if invalid
    """
    mzs = np.asarray(raw["mzs"], dtype=np.float64)
    ints = np.asarray(raw["intensities"], dtype=np.float64)

    # Quick shape check first (always needed)
    if len(mzs) == 0 or len(mzs) != len(ints):
        if metrics:
            metrics.record_integrity_error("shape_mismatch")
        return None

    # Use Rust validator if available for speed (faster than Python loops)
    if RUST_VALIDATOR_AVAILABLE:
        try:
            # Convert to float32 for Rust (matches Rust function signature)
            mzs_f32 = np.asarray(mzs, dtype=np.float32)
            ints_f32 = np.asarray(ints, dtype=np.float32)
            
            # Fast Rust validation (checks finite, positive m/z, valid precursor)
            is_valid = msms_rust.validate_spectrum(
                mzs_f32, ints_f32, float(raw["precursor_mz"])
            )
            
            if not is_valid:
                if metrics:
                    metrics.record_integrity_error("rust_validation_failed")
                return None
            
            # Additional check for max_precursor_mz (not in Rust validator)
            if not (0 < raw["precursor_mz"] < cfg.max_precursor_mz):
                if metrics:
                    metrics.record_integrity_error("invalid_precursor_mz")
                return None
        except Exception as e:
            # Fall back to Python validation if Rust fails
            if not np.isfinite(mzs).all() or not np.isfinite(ints).all():
                if metrics:
                    metrics.record_integrity_error("nonfinite_values")
                return None

            if (mzs <= 0).any():
                if metrics:
                    metrics.record_integrity_error("negative_mz")
                return None

            if not (0 < raw["precursor_mz"] < cfg.max_precursor_mz):
                if metrics:
                    metrics.record_integrity_error("invalid_precursor_mz")
                return None
    else:
        # Python validation (fallback or if Rust not available)
        if not np.isfinite(mzs).all() or not np.isfinite(ints).all():
            if metrics:
                metrics.record_integrity_error("nonfinite_values")
            return None

        if (mzs <= 0).any():
            if metrics:
                metrics.record_integrity_error("negative_mz")
            return None

        if not (0 < raw["precursor_mz"] < cfg.max_precursor_mz):
            if metrics:
                metrics.record_integrity_error("invalid_precursor_mz")
            return None

    # Soft quality filters (attrition)
    if len(mzs) < cfg.min_peaks:
        if metrics:
            metrics.record_attrition("too_few_peaks")
        return None

    # Processing
    if cfg.sort_mz:
        order = np.argsort(mzs)
        mzs = mzs[order]
        ints = ints[order]

    if cfg.normalize_intensities:
        m = ints.max()
        if m > 0:
            ints = ints / m

    mzs = mzs.astype("float32")
    ints = ints.astype("float32")

    return {
        "sample_id": sample_id,
        "mzs": mzs,
        "ints": ints,
        "precursor_mz": np.float32(raw["precursor_mz"]),
        "charge": np.int8(raw["charge"]),
        "adduct": raw["adduct"] or "unknown",
        "instrument_type": raw["instrument_type"] or "unknown",
        "collision_energy": np.float32(raw["collision_energy"]),
        "smiles": raw["smiles"],
        "inchikey": raw["inchikey"],
        "formula": raw["formula"],
    }
