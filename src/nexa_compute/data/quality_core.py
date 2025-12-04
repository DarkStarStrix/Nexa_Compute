"""Python wrapper for nexa_data_quality Rust extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import nexa_data_quality as _rust
except ImportError:
    _rust = None


class QualityCoreError(Exception):
    """Base exception for quality core errors."""
    pass


class NexaDataQuality:
    """Interface to the high-performance Rust quality engine."""

    @property
    def available(self) -> bool:
        return _rust is not None

    def _check_available(self) -> None:
        if not self.available:
            raise QualityCoreError("nexa_data_quality Rust extension not installed or failed to load.")

    def filter_batch(self, input_path: str | Path, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Filter a batch of data."""
        self._check_available()
        try:
            result_json = _rust.filter_batch(str(input_path), json.dumps(config))
            # Parse result to get output path and stats
            # Placeholder return
            return "filtered.parquet", {}
        except Exception as e:
            raise QualityCoreError(f"Filter batch failed: {e}") from e

    def deduplicate_batch(self, input_path: str | Path) -> str:
        """Deduplicate a batch of data."""
        self._check_available()
        try:
            return _rust.deduplicate_batch(str(input_path))
        except Exception as e:
            raise QualityCoreError(f"Dedup batch failed: {e}") from e


# Global instance
quality_core = NexaDataQuality()

