"""Wrapper around the OpenAI API for teacher generation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence


try:  # pragma: no cover - thin wrapper around external SDK
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - raise helpful guidance
    raise ImportError(
        "The `openai` package is required for distillation collection. "
        "Install it via `pip install openai`."
    ) from exc


@dataclass(frozen=True)
class PromptRequest:
    """A single prompt request for the teacher model."""

    user_prompt: str
    system_prompt: Optional[str] = None
    metadata: MutableMapping[str, str] | None = None


@dataclass
class PromptResult:
    """A structured response produced by the teacher model."""

    output_text: str
    latency_ms: int
    model_id: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: dict | None = None


class OpenAIClient:
    """Lightweight OpenAI API client with sane defaults for distillation."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set the `OPENAI_API_KEY` environment "
                "variable or pass `api_key` explicitly."
            )

        # Use OpenAI API (not OpenRouter)
        final_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        # Get project from environment if available (some API keys require project context)
        project_id = None
        for key in list(os.environ.keys()):
            if "OPENAI" in key.upper() and "PROJECT" in key.upper():
                project_id = os.environ.get(key)
                break
        
        # Create client with or without project
        self._client = OpenAI(
            api_key=self._api_key,
            organization=organization or os.getenv("OPENAI_ORG"),
            base_url=final_base_url,
            project=project_id,  # Use project if available - some models require it
        )

        self._default_model = default_model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p

    def generate(
        self,
        request: PromptRequest,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_params: Optional[Dict[str, object]] = None,
    ) -> PromptResult:
        """Generate a single completion for ``request``."""

        model_id = model or self._default_model
        temperature_value = temperature if temperature is not None else self._temperature
        max_tokens_value = (
            max_output_tokens if max_output_tokens is not None else self._max_output_tokens
        )
        top_p_value = top_p if top_p is not None else self._top_p

        messages: List[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        # GPT-5 models have different parameter requirements
        is_gpt5 = "gpt-5" in str(model_id).lower()
        token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"
        
        params: Dict[str, object] = {
            "model": model_id,
            "messages": messages,
        }
        
        # GPT-5 only supports temperature=1 (default), so omit temperature and top_p
        if not is_gpt5:
            params["temperature"] = temperature_value
            params["top_p"] = top_p_value
        
        params[token_param] = max_tokens_value
        
        # For GPT-5, explicitly remove temperature and top_p from extra_params if present
        if is_gpt5 and extra_params:
            extra_params = {k: v for k, v in extra_params.items() if k not in ["temperature", "top_p"]}
        
        if extra_params:
            params.update(extra_params)
        
        # Final safety check: remove temperature/top_p for GPT-5 even if they somehow got added
        if is_gpt5:
            params.pop("temperature", None)
            params.pop("top_p", None)

        start_time = time.perf_counter()
        response = self._client.chat.completions.create(**params)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        choice = response.choices[0]
        # Extract text content - handle different response formats
        text = ""
        if hasattr(choice, 'message'):
            if hasattr(choice.message, 'content'):
                text = choice.message.content or ""
            elif hasattr(choice.message, 'text'):
                text = choice.message.text or ""
        elif hasattr(choice, 'text'):
            text = choice.text or ""
        elif hasattr(choice, 'content'):
            text = choice.content or ""
        elif hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
            text = choice.delta.content or ""
        
        # Fallback: try to get from raw response
        if not text and hasattr(response, 'model_dump'):
            try:
                dump = response.model_dump()
                if 'choices' in dump and len(dump['choices']) > 0:
                    choice_data = dump['choices'][0]
                    if 'message' in choice_data and 'content' in choice_data['message']:
                        text = choice_data['message']['content'] or ""
            except:
                pass
        
        # Final fallback
        if not text:
            text = str(choice) if choice else ""

        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }

        return PromptResult(
            output_text=text,
            latency_ms=latency_ms,
            model_id=model_id,
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def generate_batch(
        self,
        requests: Sequence[PromptRequest],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_params: Optional[Dict[str, object]] = None,
    ) -> List[PromptResult]:
        """Generate completions for a batch of prompts sequentially."""

        results: List[PromptResult] = []
        for request in requests:
            results.append(
                self.generate(
                    request,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_p=top_p,
                    extra_params=extra_params,
                )
            )
        return results

