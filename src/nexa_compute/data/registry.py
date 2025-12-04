"""Dataset registry and repository."""

from __future__ import annotations

from typing import Callable, Dict

from torch.utils.data import Dataset

from .dataset import DatasetFactory, DatasetBuildContext
from ..config.schema import DataConfig


class DatasetRegistry:
    """Keeps track of available dataset builders."""

    def __init__(self) -> None:
        self._factory = DatasetFactory()

    def register(self, name: str, builder: Callable[[DatasetBuildContext], Dataset]) -> None:
        self._factory.register(name, builder)

    def build(self, config: DataConfig, *, split: str) -> Dataset:
        """Build a dataset from configuration.
        
        **Versioned Manifest**: All datasets must have a dataset_version specified in config.
        This version is used for reproducibility and manifest tracking.
        """
        if not config.dataset_version:
            LOGGER.warning(
                "dataset_version_missing",
                extra={"dataset_name": config.dataset_name},
            )
        return self._factory.build(config, split=split)

    def available(self) -> Dict[str, Callable[[DatasetBuildContext], Dataset]]:
        return dict(self._factory._builders)


DEFAULT_REGISTRY = DatasetRegistry()
