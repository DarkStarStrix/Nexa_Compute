"""Common exception hierarchy used across NexaCompute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class NexaError(RuntimeError):
    message: str
    code: str = "nexa_error"
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__init__(self.message)
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "metadata": self.metadata}


class RetryableError(NexaError):
    code = "retryable_error"


class NonRetryableError(NexaError):
    code = "non_retryable_error"


class ResourceError(NexaError):
    code = "resource_error"


