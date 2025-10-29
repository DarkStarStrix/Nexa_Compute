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
        return self._factory.build(config, split=split)

    def available(self) -> Dict[str, Callable[[DatasetBuildContext], Dataset]]:
        return dict(self._factory._builders)


DEFAULT_REGISTRY = DatasetRegistry()
