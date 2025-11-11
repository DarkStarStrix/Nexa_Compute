"""Prompt assets for tool-using fine-tuning."""

from __future__ import annotations

from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROMPT_ROOT / "system_toolproto.txt"
FEW_SHOTS_PATH = PROMPT_ROOT / "few_shots_toolproto.jsonl"

__all__ = ["PROMPT_ROOT", "SYSTEM_PROMPT_PATH", "FEW_SHOTS_PATH"]

