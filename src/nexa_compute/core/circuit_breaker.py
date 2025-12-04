"""Circuit breaker implementation for protecting flaky dependencies."""

from __future__ import annotations

import time
from dataclasses import dataclass


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the circuit breaker is open and rejecting calls."""


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")
        self._state = "closed"
        self._failure_count = 0
        self._last_failure_timestamp = 0.0
        self._half_open_attempts = 0

    def before_call(self) -> None:
        """Check current state and raise if breaker is open."""
        if self._state == "open":
            if time.time() - self._last_failure_timestamp >= self.recovery_timeout:
                self._state = "half_open"
                self._half_open_attempts = 0
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        if self._state == "half_open":
            if self._half_open_attempts >= self.half_open_max_calls:
                raise CircuitBreakerOpenError("Circuit breaker half-open trial limit reached")
            self._half_open_attempts += 1

    def on_success(self) -> None:
        """Reset breaker after a successful call."""
        self._failure_count = 0
        self._half_open_attempts = 0
        self._state = "closed"

    def on_failure(self, _: BaseException) -> None:
        """Record failure and open breaker if needed."""
        if self._state == "half_open":
            self._trip()
            return

        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = "open"
        self._last_failure_timestamp = time.time()
        self._failure_count = 0
        self._half_open_attempts = 0

    @property
    def state(self) -> str:
        return self._state

