"""Model registry and reference architectures."""

from .base import DEFAULT_MODEL_REGISTRY, ModelBuilder, ModelRegistry
from .mlp import MLPClassifier
from .resnet import ResNetClassifier
from .transformer import TransformerEncoderClassifier

__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "ModelBuilder",
    "ModelRegistry",
    "MLPClassifier",
    "ResNetClassifier",
    "TransformerEncoderClassifier",
]
