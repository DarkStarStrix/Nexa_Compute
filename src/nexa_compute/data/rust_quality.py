"""Python wrapper for the Rust-powered data quality core."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import nexa_data_quality as rust_quality
except ImportError:
    rust_quality = None

class RustQualityError(ImportError):
    """Raised when the Rust quality extension is missing."""
    pass

def _ensure_rust_quality():
    if rust_quality is None:
        raise RustQualityError(
            "The 'nexa_data_quality' Rust extension is not installed. "
            "Please build it using `maturin develop` or `pip install .` from the rust/nexa_data_quality directory."
        )

@dataclass
class FilterConfig:
    text_column: str
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    bad_patterns: Optional[List[str]] = None
    dedup_enabled: bool = False

    def to_rust(self) -> Any:
        _ensure_rust_quality()
        return rust_quality.FilterConfig(
            self.text_column,
            self.min_length,
            self.max_length,
            self.bad_patterns or [],
            self.dedup_enabled
        )

def filter_corpus(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    rejected_path: Union[str, Path],
    config: FilterConfig
) -> Dict[str, Any]:
    """
    Filter a corpus using the Rust quality engine.
    
    Args:
        input_path: Path to input parquet file or directory.
        output_path: Path to output parquet file or directory.
        rejected_path: Path to rejected parquet file or directory.
        config: Filter configuration.
        
    Returns:
        Dictionary containing filter statistics.
    """
    _ensure_rust_quality()
    rust_config = config.to_rust()
    stats = rust_quality.filter_batch(
        str(input_path),
        str(output_path),
        str(rejected_path),
        rust_config
    )
    
    return {
        "total": stats.total,
        "kept": stats.kept,
        "rejected_length": stats.rejected_length,
        "rejected_pattern": stats.rejected_pattern,
        "rejected_dedup": stats.rejected_dedup,
        "length_histogram": stats.length_histogram
    }

