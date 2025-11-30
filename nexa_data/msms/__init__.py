"""MS/MS data processing pipeline."""

from .config import PipelineConfig, load_config
from .hdf5_reader import HDF5SpectrumSource
from .manifest import build_dataset_manifest
from .metrics import PipelineMetrics
from .processor import BatchProcessor
from .shard_writer import ShardWriter
from .transforms import clean_and_canonicalize
from .validate import generate_quality_report, test_determinism, validate_shards

__all__ = [
    "PipelineConfig",
    "load_config",
    "HDF5SpectrumSource",
    "build_dataset_manifest",
    "PipelineMetrics",
    "BatchProcessor",
    "ShardWriter",
    "clean_and_canonicalize",
    "generate_quality_report",
    "test_determinism",
    "validate_shards",
]

