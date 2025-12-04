"""
Retry helpers with exponential backoff, jitter, and optional circuit breaker hooks.

The module exposes both decorator-based and direct-call helpers so that call sites
can opt-in without restructuring their logic. The implementation is intentionally
dependency-free so it can be shared by CLI tools, API handlers, and worker code.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional, Protocol, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Awaitable[Any]])


class CircuitBreakerLike(Protocol):
    """Minimal surface required from a circuit breaker instance."""

    def before_call(self) -> None: ...

    def on_success(self) -> None: ...

    def on_failure(self, exc: BaseException) -> None: ...


@dataclass
class RetryPolicy:
    """Configuration for retry attempts."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.2
    retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,)
    non_retry_exceptions: Tuple[Type[BaseException], ...] = ()
    respect_retry_after: bool = True
    timeout: Optional[float] = None
    give_up_exceptions: Tuple[Type[BaseException], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be > 0")
        if self.max_delay < self.base_delay:
            self.max_delay = self.base_delay
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, last_exception: BaseException, attempts: int) -> None:
        super().__init__(f"Operation failed after {attempts} attempts: {last_exception}")
        self.last_exception = last_exception
        self.attempts = attempts


def _compute_delay(policy: RetryPolicy, attempt: int, retry_after: Optional[float] = None) -> float:
    """Compute exponential backoff delay with optional jitter."""
    delay = min(policy.base_delay * (2 ** (attempt - 1)), policy.max_delay)
    if retry_after is not None:
        delay = max(delay, retry_after)
    if policy.jitter:
        jitter_amount = delay * policy.jitter
        delay = delay - jitter_amount + random.uniform(0, jitter_amount * 2)
    return max(delay, 0.0)


def _should_retry(exc: BaseException, policy: RetryPolicy, is_retryable: Optional[Callable[[BaseException], bool]]) -> bool:
    if isinstance(exc, policy.give_up_exceptions):
        return False
    if isinstance(exc, policy.non_retry_exceptions):
        return False
    if not isinstance(exc, policy.retry_exceptions):
        return False
    if is_retryable is not None and not is_retryable(exc):
        return False
    return True


def _extract_retry_after(exc: BaseException) -> Optional[float]:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (ValueError, TypeError):
        return None


def retry_call(
    func: Callable[..., Any],
    *args: Any,
    policy: RetryPolicy,
    circuit_breaker: Optional[CircuitBreakerLike] = None,
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> Any:
    """Execute ``func`` with retry semantics."""
    if circuit_breaker:
        circuit_breaker.before_call()

    attempt = 0
    while True:
        attempt += 1
        try:
            result = func(*args, **kwargs)
            if circuit_breaker:
                circuit_breaker.on_success()
            return result
        except BaseException as exc:
            if circuit_breaker:
                circuit_breaker.on_failure(exc)
            if attempt >= policy.max_attempts or not _should_retry(exc, policy, is_retryable):
                raise RetryError(exc, attempt) from exc
            delay = _compute_delay(policy, attempt, _extract_retry_after(exc) if policy.respect_retry_after else None)
            if on_retry:
                on_retry(attempt, exc, delay)
            time.sleep(delay)


async def async_retry_call(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    policy: RetryPolicy,
    circuit_breaker: Optional[CircuitBreakerLike] = None,
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> Any:
    """Async equivalent of :func:`retry_call`."""
    if circuit_breaker:
        circuit_breaker.before_call()

    attempt = 0
    while True:
        attempt += 1
        try:
            result = await func(*args, **kwargs)
            if circuit_breaker:
                circuit_breaker.on_success()
            return result
        except BaseException as exc:
            if circuit_breaker:
                circuit_breaker.on_failure(exc)
            if attempt >= policy.max_attempts or not _should_retry(exc, policy, is_retryable):
                raise RetryError(exc, attempt) from exc
            delay = _compute_delay(policy, attempt, _extract_retry_after(exc) if policy.respect_retry_after else None)
            if on_retry:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)


def retry(
    *,
    policy: Optional[RetryPolicy] = None,
    circuit_breaker: Optional[CircuitBreakerLike] = None,
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[[F], F]:
    """Decorator applying retry logic to synchronous callables."""

    def decorator(func: F) -> F:
        effective_policy = policy or RetryPolicy()

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return retry_call(
                func,
                *args,
                policy=effective_policy,
                circuit_breaker=circuit_breaker,
                is_retryable=is_retryable,
                on_retry=on_retry,
                **kwargs,
            )

        return wrapped  # type: ignore[return-value]

    return decorator


def async_retry(
    *,
    policy: Optional[RetryPolicy] = None,
    circuit_breaker: Optional[CircuitBreakerLike] = None,
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[[AF], AF]:
    """Decorator applying retry logic to async callables."""

    def decorator(func: AF) -> AF:
        effective_policy = policy or RetryPolicy()

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            return await async_retry_call(
                func,
                *args,
                policy=effective_policy,
                circuit_breaker=circuit_breaker,
                is_retryable=is_retryable,
                on_retry=on_retry,
                **kwargs,
            )

        return wrapped  # type: ignore[return-value]

    return decorator


def make_linearized_delays(policy: RetryPolicy, attempts: int) -> Iterable[float]:
    """Expose delays for testing purposes."""
    for attempt in range(1, attempts + 1):
        yield _compute_delay(policy, attempt)


__all__ = [
    "RetryPolicy",
    "RetryError",
    "retry",
    "async_retry",
    "retry_call",
    "async_retry_call",
    "make_linearized_delays",
    "CircuitBreakerLike",
]

