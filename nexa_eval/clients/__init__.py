"""Client abstractions for external services used in evaluations."""

from .local_api import (
    LocalAPIConfig,
    LocalGenerationRequest,
    LocalGenerationResponse,
    LocalInferenceClient,
)
from .openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterRequest,
    OpenRouterResponse,
)

__all__ = [
    "LocalAPIConfig",
    "LocalGenerationRequest",
    "LocalGenerationResponse",
    "LocalInferenceClient",
    "OpenRouterClient",
    "OpenRouterConfig",
    "OpenRouterRequest",
    "OpenRouterResponse",
]

