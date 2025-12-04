"""Data ingestion, preprocessing, and loading.

Responsibility: Manages dataset lifecycle including ingestion, versioning, quality filtering, 
statistical analysis, and integration with Rust-powered preprocessing engines.
"""

from .dataset import DatasetFactory, SyntheticClassificationDataset
from .pipeline import DataPipeline
from .registry import DatasetRegistry, DEFAULT_REGISTRY

__all__ = [
    "DatasetFactory",
    "SyntheticClassificationDataset",
    "DataPipeline",
    "DatasetRegistry",
    "DEFAULT_REGISTRY",
]
