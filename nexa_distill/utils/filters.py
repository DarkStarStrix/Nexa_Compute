"""Heuristic filters for teacher outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

from .texttools import estimate_token_count, normalize_whitespace


ACTION_VERBS = {
    "analyze",
    "assess",
    "calibrate",
    "compare",
    "culture",
    "evaluate",
    "incubate",
    "measure",
    "mix",
    "prepare",
    "run",
    "simulate",
    "synthesize",
    "test",
    "validate",
}

_CITATION_PATTERN = re.compile(r"(\[[0-9]{1,3}\]|\(ref\)|doi:\S+|arxiv:\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for heuristic filtering."""

    min_char_length: int = 120
    min_token_length: int = 80
    require_action_verb: bool = True
    ban_citations: bool = True
    action_verbs: tuple[str, ...] | None = None


def has_action_verbs(text: str, verbs: Iterable[str] | None = None) -> bool:
    """Return ``True`` if any action verb is present in ``text``."""

    verbs_to_use = set(v.lower() for v in (verbs or ACTION_VERBS))
    text_lower = normalize_whitespace(text).lower()
    return any(re.search(rf"\b{re.escape(verb)}\b", text_lower) for verb in verbs_to_use)


def contains_citation_markers(text: str) -> bool:
    """Return ``True`` when citation-like patterns are detected."""

    return bool(_CITATION_PATTERN.search(text))


def basic_filters(
    record: Mapping[str, str],
    *,
    config: FilterConfig | None = None,
) -> MutableMapping[str, bool]:
    """Evaluate heuristic filters and return flag outcomes."""

    cfg = config or FilterConfig()
    output = normalize_whitespace(record.get("teacher_output", ""))

    flags: MutableMapping[str, bool] = {}

    flags["length_char_ok"] = len(output) >= cfg.min_char_length
    flags["length_token_ok"] = estimate_token_count(output) >= cfg.min_token_length
    if cfg.require_action_verb:
        verbs = cfg.action_verbs if cfg.action_verbs is not None else None
        flags["action_verb_ok"] = has_action_verbs(output, verbs=verbs)
    else:
        flags["action_verb_ok"] = True
    if cfg.ban_citations:
        flags["no_citations"] = not contains_citation_markers(output)
    else:
        flags["no_citations"] = True

    flags["passes_all"] = all(flags.values())
    return flags

