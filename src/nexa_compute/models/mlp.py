"""Multi-layer perceptron reference model."""

from __future__ import annotations

from typing import List

import torch.nn as nn

from ..config.schema import ModelConfig
from .base import DEFAULT_MODEL_REGISTRY


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):  # type: ignore[override]
        return self.network(inputs)


def build_mlp(config: ModelConfig) -> nn.Module:
    params = config.parameters
    input_dim = int(params.get("input_dim", 32))
    hidden_dims = params.get("hidden_dims", [128, 64])
    num_classes = int(params.get("num_classes", 2))
    dropout = float(params.get("dropout", 0.1))
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
    )
    return model


DEFAULT_MODEL_REGISTRY.register("mlp_classifier", build_mlp)
