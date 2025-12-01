import time
import warnings

# Suppress upstream PyTorch/pynvml FutureWarning until dependency updates land.
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)

import pytest

from nexa_compute.utils import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    ExecutionTimeoutError,
    RetryPolicy,
    execution_timeout,
    retry_call,
)
from nexa_compute.utils.secrets import SecretManager, SecretsConfig
from nexa_tools.sandbox import SandboxRunner


def test_retry_call_succeeds_after_failures(monkeypatch):
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("transient")
        return "ok"

    monkeypatch.setattr(time, "sleep", lambda _: None)
    result = retry_call(flaky, policy=RetryPolicy(max_attempts=5, retry_exceptions=(ValueError,)))
    assert result == "ok"
    assert attempts["count"] == 3


def test_circuit_breaker_transitions():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    breaker.before_call()
    breaker.on_failure(RuntimeError("fail"))
    breaker.before_call()
    breaker.on_failure(RuntimeError("fail"))
    with pytest.raises(CircuitBreakerOpenError):
        breaker.before_call()
    time.sleep(0.11)
    breaker.before_call()
    breaker.on_success()


def test_secret_manager_env_backend(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_TOKEN", "super-secret")
    config = SecretsConfig(backend="env", env_prefix="TEST_")
    manager = SecretManager(config)
    assert manager.get("SECRET_TOKEN") == "super-secret"


def test_execution_timeout_expires():
    with pytest.raises(ExecutionTimeoutError):
        with execution_timeout(0.1):
            time.sleep(0.2)


def test_sandbox_validator_blocks_dangerous_import():
    runner = SandboxRunner(use_docker=False)
    with pytest.raises(ValueError):
        runner._validate_source("import os\nprint('hi')")  # type: ignore[attr-defined]

    runner._validate_source("import math\nprint(math.sqrt(9))")  # type: ignore[attr-defined]

