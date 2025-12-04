"""Convenience client for interacting with the OpenRouter API."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import BaseException, Dict, Iterable, List, Optional, Sequence

import requests

from nexa_compute.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from nexa_compute.core.exceptions import NexaError, NonRetryableError, RetryableError
from nexa_compute.core.retry import RetryPolicy, retry_call

LOGGER = logging.getLogger(__name__)


class OpenRouterError(NexaError):
    """Base class for OpenRouter failures."""


class OpenRouterTransientError(OpenRouterError, RetryableError):
    """Retryable OpenRouter error carrying optional retry-after hint."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OpenRouterFatalError(OpenRouterError, NonRetryableError):
    """Non-retryable OpenRouter error."""


@dataclass(frozen=True)
class OpenRouterConfig:
    """Configuration for the OpenRouter client."""

    model: str
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: int = 60
    max_retries: int = 3
    retry_backoff: float = 1.0
    retry_jitter: float = 0.2
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 60.0
    circuit_breaker_half_open_max_calls: int = 1
    default_temperature: float = 0.2
    default_max_tokens: int = 2048
    headers: Dict[str, str] = field(default_factory=dict)

    def resolved_api_key(self) -> str:
        """Resolve API key, falling back to environment variable."""

        if self.api_key:
            return self.api_key
        env_key = os.getenv("OPENROUTER_API_KEY")
        if not env_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Export it or pass api_key to OpenRouterConfig."
            )
        return env_key


@dataclass
class OpenRouterRequest:
    """Single completion request payload."""

    prompt: str
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, object]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class OpenRouterUsage:
    """Tracks token usage for cost estimation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class OpenRouterResponse:
    """Structured response containing model output and metadata."""

    prompt: str
    output_text: str
    model: str
    latency_ms: float
    usage: OpenRouterUsage
    raw: Dict[str, object] = field(default_factory=dict)
    metadata: Optional[Dict[str, object]] = None


class OpenRouterClient:
    """Wrapper around the OpenRouter completion endpoint."""

    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._config.resolved_api_key()}",
                "Content-Type": "application/json",
            }
        )
        self._session.headers.update(config.headers)
        self._retry_policy = RetryPolicy(
            max_attempts=max(1, config.max_retries),
            base_delay=max(0.1, config.retry_backoff),
            max_delay=max(0.1, config.retry_backoff * 8),
            jitter=min(max(config.retry_jitter, 0.0), 1.0),
            retry_exceptions=(requests.RequestException, OpenRouterTransientError),
            non_retry_exceptions=(OpenRouterFatalError,),
        )
        self._breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_seconds,
            half_open_max_calls=config.circuit_breaker_half_open_max_calls,
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self._session.close()

    def generate(
        self,
        requests_: Sequence[OpenRouterRequest],
        *,
        model: Optional[str] = None,
        batch_size: int = 1,
    ) -> List[OpenRouterResponse]:
        """Generate completions for the supplied prompts."""

        results: List[OpenRouterResponse] = []
        for chunk in _chunk(requests_, batch_size=batch_size):
            for request in chunk:
                results.append(self._execute_request(request, model=model))
        return results

    def _execute_request(
        self,
        request: OpenRouterRequest,
        *,
        model: Optional[str] = None,
    ) -> OpenRouterResponse:
        payload = self._build_payload(request, model=model)
        start = time.perf_counter()

        def _call() -> OpenRouterResponse:
            response = self._session.post(
                f"{self._config.base_url}/chat/completions",
                json=payload,
                timeout=self._config.timeout_s,
            )
            if response.status_code != 200:
                self._raise_for_status(response)

            latency_ms = (time.perf_counter() - start) * 1000.0
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            output_text = message.get("content") or ""
            finish_reason = choice.get("finish_reason", "unknown")

            if not output_text:
                import warnings

                warnings.warn(
                    f"Empty response from {payload.get('model', 'unknown')}. "
                    f"Finish reason: {finish_reason}. "
                    f"Prompt tokens: {body.get('usage', {}).get('prompt_tokens', 0)}",
                    stacklevel=2,
                )
            elif finish_reason == "length":
                import warnings

                warnings.warn(
                    f"Response truncated (max_tokens reached) for {payload.get('model', 'unknown')}. "
                    f"Output length: {len(output_text)}",
                    stacklevel=2,
                )

            usage_info = body.get("usage", {})
            usage = OpenRouterUsage(
                prompt_tokens=int(usage_info.get("prompt_tokens", 0)),
                completion_tokens=int(usage_info.get("completion_tokens", 0)),
            )
            metadata = request.metadata or {}
            return OpenRouterResponse(
                prompt=request.prompt,
                output_text=output_text,
                model=body.get("model", payload["model"]),
                latency_ms=latency_ms,
                usage=usage,
                raw=body,
                metadata=metadata if metadata else None,
            )

        try:
            return retry_call(
                _call,
                policy=self._retry_policy,
                on_retry=self._log_retry_attempt,
                circuit_breaker=self._breaker,
            )
        except CircuitBreakerOpenError as exc:
            raise OpenRouterError("OpenRouter circuit breaker is open; refusing to execute request.") from exc

    def _log_retry_attempt(self, attempt: int, exc: BaseException, delay: float) -> None:
        LOGGER.warning(
            "openrouter_retry_attempt",
            extra={
                "extra_context": {
                    "attempt": attempt,
                    "delay_s": round(delay, 3),
                    "exc": repr(exc),
                }
            },
        )

    def _raise_for_status(self, response: requests.Response) -> None:
        retry_after = response.headers.get("Retry-After")
        retry_after_value: Optional[float] = None
        if retry_after is not None:
            try:
                retry_after_value = float(retry_after)
            except ValueError:
                retry_after_value = None

        status = response.status_code
        body = response.text

        if status == 429 or 500 <= status < 600:
            raise OpenRouterTransientError(
                f"OpenRouter transient failure {status}: {body}",
                retry_after=retry_after_value,
            )

        if 400 <= status < 500:
            raise OpenRouterFatalError(f"OpenRouter request failed {status}: {body}")

        response.raise_for_status()

    def _build_payload(
        self,
        request: OpenRouterRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": model or self._config.model,
            "temperature": request.temperature
            if request.temperature is not None
            else self._config.default_temperature,
            "max_tokens": request.max_tokens
            if request.max_tokens is not None
            else self._config.default_max_tokens,
            "messages": _format_messages(request.prompt, request.system_prompt),
        }
        if request.metadata:
            payload["metadata"] = request.metadata
        return payload


def _chunk(iterable: Sequence[OpenRouterRequest], batch_size: int) -> Iterable[Sequence[OpenRouterRequest]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for index in range(0, len(iterable), batch_size):
        yield iterable[index : index + batch_size]


def _format_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages

