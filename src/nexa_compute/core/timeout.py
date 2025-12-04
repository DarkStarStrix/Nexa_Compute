"""Timeout helpers for blocking sections."""

from __future__ import annotations

import signal
import time
from contextlib import contextmanager
from typing import Iterator, Optional


class ExecutionTimeoutError(TimeoutError):
    """Raised when execution exceeds the configured timeout."""


@contextmanager
def execution_timeout(seconds: Optional[float], *, message: str = "Operation timed out") -> Iterator[None]:
    """Context manager that raises if the block exceeds ``seconds`` wall-clock time."""
    if not seconds or seconds <= 0:
        yield
        return

    if hasattr(signal, "SIGALRM"):
        def _handler(signum, frame):  # pragma: no cover - depends on OS
            raise ExecutionTimeoutError(message)

        previous = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    else:
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        if elapsed > seconds:
            raise ExecutionTimeoutError(message)

