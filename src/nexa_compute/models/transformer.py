"""Lightweight Transformer encoder classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..config.schema import ModelConfig
from .base import DEFAULT_MODEL_REGISTRY


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, num_layers: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, ff_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=ff_dim, nhead=num_heads, dim_feedforward=ff_dim * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(ff_dim, num_classes)

    def forward(self, inputs):  # type: ignore[override]
        # Expect shape (batch, seq_len, input_dim)
        embedded = self.embedding(inputs)
        encoded = self.encoder(embedded)
        pooled = self.pool(encoded.transpose(1, 2)).squeeze(-1)
        return self.classifier(pooled)


def build_transformer(config: ModelConfig) -> nn.Module:
    params: dict[str, Any] = dict(config.parameters)
    input_dim = int(params.pop("input_dim", 64))
    num_heads = int(params.pop("num_heads", 4))
    ff_dim = int(params.pop("ff_dim", 128))
    num_layers = int(params.pop("num_layers", 2))
    num_classes = int(params.pop("num_classes", 2))
    dropout = float(params.pop("dropout", 0.1))
    model = TransformerEncoderClassifier(
        input_dim=input_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
    return model


DEFAULT_MODEL_REGISTRY.register("transformer_classifier", build_transformer)
