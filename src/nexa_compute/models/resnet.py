"""Wrapper around torchvision ResNet architectures."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..config.schema import ModelConfig
from .base import DEFAULT_MODEL_REGISTRY


class ResNetClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, inputs):  # type: ignore[override]
        return self.backbone(inputs)


def build_resnet(config: ModelConfig) -> nn.Module:
    from torchvision import models

    params: dict[str, Any] = dict(config.parameters)
    variant = params.pop("variant", "resnet18")
    pretrained = bool(config.pretrained or params.pop("pretrained", False))
    num_classes = int(params.pop("num_classes", 2))
    constructor = getattr(models, variant)
    backbone = constructor(weights="DEFAULT" if pretrained else None)
    return ResNetClassifier(backbone, num_classes)


DEFAULT_MODEL_REGISTRY.register("resnet_classifier", build_resnet)
