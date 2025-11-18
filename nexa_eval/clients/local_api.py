"""Client for calling the local Nexa inference FastAPI endpoint."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import requests


class LocalInferenceError(RuntimeError):
    """Raised when the local inference server responds with an error status."""


@dataclass(frozen=True)
class LocalAPIConfig:
    """Configuration for the local inference API client."""

    base_url: str = "http://127.0.0.1:8000"
    infer_endpoint: str = "/infer"
    timeout_s: int = 60
    default_temperature: float = 0.2
    default_max_tokens: int = 512
    headers: Dict[str, str] = field(default_factory=dict)

    def infer_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.infer_endpoint}"


@dataclass
class LocalGenerationRequest:
    """Single generation request targeting the local API."""

    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, object]] = None


@dataclass
class LocalGenerationResponse:
    """Structured result returned by the local inference client."""

    prompt: str
    output_text: str
    latency_ms: float
    tokens: int
    model_id: str
    metadata: Optional[Dict[str, object]] = None
    raw: Dict[str, object] = field(default_factory=dict)


class LocalInferenceClient:
    """Wrapper around the Nexa FastAPI inference server."""

    def __init__(self, config: Optional[LocalAPIConfig] = None) -> None:
        self._config = config or LocalAPIConfig()
        self._session = requests.Session()
        if self._config.headers:
            self._session.headers.update(self._config.headers)

    def close(self) -> None:
        self._session.close()

    def generate(
        self,
        requests_: Sequence[LocalGenerationRequest],
        *,
        batch_size: int = 1,
    ) -> List[LocalGenerationResponse]:
        results: List[LocalGenerationResponse] = []
        for chunk in _chunk(requests_, batch_size=batch_size):
            for request in chunk:
                results.append(self._execute_request(request))
        return results

    def _execute_request(self, request: LocalGenerationRequest) -> LocalGenerationResponse:
        payload = {
            "prompt": _combine_prompt(request.prompt, request.system_prompt),
            "temperature": request.temperature
            if request.temperature is not None
            else self._config.default_temperature,
            "max_tokens": request.max_tokens
            if request.max_tokens is not None
            else self._config.default_max_tokens,
        }
        if request.metadata:
            payload["metadata"] = request.metadata

        start = time.perf_counter()
        response = self._session.post(
            self._config.infer_url(),
            json=payload,
            timeout=self._config.timeout_s,
        )
        if response.status_code != 200:
            raise LocalInferenceError(
                f"Local inference request failed with status {response.status_code}: {response.text}"
            )

        body = response.json()
        latency_ms = (time.perf_counter() - start) * 1000.0
        metadata = request.metadata or {}
        return LocalGenerationResponse(
            prompt=request.prompt,
            output_text=body.get("text", ""),
            latency_ms=float(body.get("latency_ms", latency_ms)),
            tokens=int(body.get("tokens", 0)),
            model_id=body.get("model_id", "local"),
            metadata=metadata if metadata else None,
            raw=body,
        )


def _combine_prompt(user_prompt: str, system_prompt: Optional[str]) -> str:
    if not system_prompt:
        return user_prompt
    system_prompt = system_prompt.strip()
    if not system_prompt.endswith("\n"):
        system_prompt = f"{system_prompt}\n"
    return f"{system_prompt}\n{user_prompt}"


def _chunk(iterable: Sequence[LocalGenerationRequest], batch_size: int) -> Iterable[Sequence[LocalGenerationRequest]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for index in range(0, len(iterable), batch_size):
        yield iterable[index : index + batch_size]

