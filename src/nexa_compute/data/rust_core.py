"""Python wrapper for nexa_data_core Rust extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import nexa_data_core as _rust
except ImportError:
    _rust = None


class RustCoreError(Exception):
    """Base exception for Rust core errors."""
    pass


class NexaDataCore:
    """Interface to the high-performance Rust data engine."""

    @property
    def available(self) -> bool:
        return _rust is not None

    def _check_available(self) -> None:
        if not self.available:
            raise RustCoreError("nexa_data_core Rust extension not installed or failed to load.")

    def convert_csv_to_parquet(self, input_path: str | Path, output_path: str | Path, batch_size: int = 1024) -> None:
        """Convert CSV file to Parquet using streaming transform."""
        self._check_available()
        try:
            _rust.convert_csv_to_parquet(str(input_path), str(output_path), batch_size)
        except Exception as e:
            raise RustCoreError(f"Failed to convert CSV: {e}") from e

    def shuffle_and_split(self, num_items: int, weights: List[float], seed: int = 42) -> List[List[int]]:
        """Deterministically shuffle and split indices."""
        self._check_available()
        try:
            return _rust.shuffle_and_split(num_items, weights, seed)
        except Exception as e:
            raise RustCoreError(f"Failed to shuffle and split: {e}") from e

    def parallel_process_files(self, files: List[str | Path]) -> None:
        """Process a list of files in parallel."""
        self._check_available()
        file_strs = [str(f) for f in files]
        try:
            _rust.parallel_process_files(file_strs)
        except Exception as e:
            raise RustCoreError(f"Parallel processing failed: {e}") from e

    def compute_stats(self, file_path: str | Path) -> Dict[str, Any]:
        """Compute statistics for a dataset file."""
        self._check_available()
        try:
            json_str = _rust.compute_stats_json(str(file_path))
            return json.loads(json_str)
        except Exception as e:
            raise RustCoreError(f"Stats computation failed: {e}") from e


# Global instance
rust_core = NexaDataCore()
