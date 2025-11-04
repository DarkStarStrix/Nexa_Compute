"""Text utility helpers for the distillation pipeline."""

from __future__ import annotations

import math
import re
from typing import Iterable, List


try:  # pragma: no cover - optional dependency
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]


DEFAULT_MODEL = "gpt-4o-mini"
TOKEN_FALLBACK_DIVISOR = 4


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace for consistent downstream handling."""

    return re.sub(r"\s+", " ", text).strip()


def estimate_token_count(text: str, model: str = DEFAULT_MODEL) -> int:
    """Estimate the token count for ``text`` with a best-effort fallback."""

    if not text:
        return 0

    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass

    return max(1, math.ceil(len(text) / TOKEN_FALLBACK_DIVISOR))


def chunk_text(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
    model: str = DEFAULT_MODEL,
) -> List[str]:
    """Split ``text`` into approximate token chunks for teacher prompts."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")

    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    if tiktoken is None:
        approx_chunk_chars = max_tokens * TOKEN_FALLBACK_DIVISOR
        stride = max(1, approx_chunk_chars - overlap_tokens * TOKEN_FALLBACK_DIVISOR)
        return [normalized[i : i + approx_chunk_chars] for i in range(0, len(normalized), stride)]

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(normalized)
    chunks: List[str] = []
    start = 0
    stride = max(1, max_tokens - overlap_tokens)
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += stride
    return chunks

