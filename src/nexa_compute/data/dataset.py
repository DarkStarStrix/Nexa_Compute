"""Dataset definitions and factories."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import Dataset

from ..config.schema import DataConfig

Tensor = torch.Tensor


class SyntheticClassificationDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Generates synthetic samples for smoke testing pipelines."""

    def __init__(self, num_samples: int = 1024, num_features: int = 32, num_classes: int = 2, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_samples, num_features, generator=generator)
        logits = torch.randn(num_samples, num_classes, generator=generator)
        self.labels = torch.argmax(logits, dim=1)

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]


@dataclass
class DatasetBuildContext:
    config: DataConfig
    split: str
    cache_key: str


class DatasetFactory:
    """Factory responsible for constructing torch datasets based on configuration."""

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[DatasetBuildContext], Dataset]] = {
            "synthetic": self._build_synthetic,
        }

    def register(self, name: str, builder: Callable[[DatasetBuildContext], Dataset]) -> None:
        self._builders[name] = builder

    def build(self, config: DataConfig, *, split: str) -> Dataset:
        if config.dataset_name not in self._builders:
            raise KeyError(f"Unknown dataset '{config.dataset_name}'. Registered: {list(self._builders)}")
        cache_key = self._cache_key(config, split)
        context = DatasetBuildContext(config=config, split=split, cache_key=cache_key)
        builder = self._builders[config.dataset_name]
        return builder(context)

    @staticmethod
    def _cache_key(config: DataConfig, split: str) -> str:
        payload = f"{config.dataset_name}|{config.dataset_version}|{split}|{config.preprocessing}|{config.augmentations}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_synthetic(context: DatasetBuildContext) -> Dataset:
        ratio_map = {
            "train": context.config.split.train,
            "validation": context.config.split.validation,
            "test": context.config.split.test,
        }
        total_samples = 4096
        split_ratio = ratio_map.get(context.split, 0.33)
        num_samples = max(32, math.floor(total_samples * split_ratio))
        params = context.config.preprocessing
        num_features = int(params.get("num_features", 32))
        num_classes = int(params.get("num_classes", 2))
        seed = int(params.get("seed", 42))
        return SyntheticClassificationDataset(
            num_samples=num_samples,
            num_features=num_features,
            num_classes=num_classes,
            seed=seed,
        )
