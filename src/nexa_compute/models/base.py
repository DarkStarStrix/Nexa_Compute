"""Base model registry primitives."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch.nn as nn

from ..config.schema import ModelConfig

ModelBuilder = Callable[[ModelConfig], nn.Module]


class ModelRegistry:
    def __init__(self) -> None:
        self._builders: Dict[str, ModelBuilder] = {}

    def register(self, name: str, builder: ModelBuilder) -> None:
        self._builders[name] = builder

    def build(self, config: ModelConfig) -> nn.Module:
        if config.name not in self._builders:
            raise KeyError(f"Unknown model '{config.name}'. Registered: {list(self._builders)}")
        return self._builders[config.name](config)

    def available(self) -> Dict[str, ModelBuilder]:
        return dict(self._builders)


DEFAULT_MODEL_REGISTRY = ModelRegistry()
