"""Utilities for constructing evaluation prompt parquets from existing data.

This module extracts prompts from distillation datasets and materialises them
into the schema required by the scientific evaluation pipeline:

    id: int64
    domain: string
    task_type: string
    difficulty: int32
    prompt: string
    seed_metadata: json (stored as dict before parquet serialisation)

Usage
-----
python -m nexa_eval.prepare_prompts \
    --source data/processed/scientific_assistant/distillation/teacher_inputs \
    --output data/processed/evaluation/prompts/prompts.parquet \
    --total 400 \
    --per-domain 80
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_OUTPUT = Path("data/processed/evaluation/prompts/prompts.parquet")
DEFAULT_SOURCE = Path("data/processed/scientific_assistant/distillation/teacher_inputs")
DEFAULT_TOTAL = 400
DEFAULT_PER_DOMAIN = 80


@dataclass(frozen=True)
class PromptRecord:
    """Structured representation of a prepared prompt."""

    id: int
    domain: str
    task_type: str
    difficulty: int
    prompt: str
    seed_metadata: Dict[str, object]


def _collect_source_files(source: Path) -> List[Path]:
    if source.is_file() and source.suffix == ".parquet":
        return [source]
    return sorted(source.glob("*.parquet"))


def _load_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["__source_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No parquet files discovered for prompt preparation.")
    combined = pd.concat(frames, ignore_index=True)
    if "user_prompt" not in combined.columns:
        raise ValueError("Expected column 'user_prompt' missing from source data.")
    return combined


def _normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={"template_name": "task_type"})
    renamed["task_type"] = renamed["task_type"].astype(str)
    renamed["domain"] = renamed["domain"].astype(str)
    renamed["prompt"] = renamed["user_prompt"].astype(str)
    renamed["seed_metadata"] = renamed.apply(
        lambda row: {
            "template_prompt": row.get("template_prompt"),
            "system_prompt": row.get("system_prompt"),
            "source_file": row.get("source_file"),
            "source_path": row.get("__source_path"),
            "generated_at": row.get("generated_at"),
        },
        axis=1,
    )
    return renamed[["domain", "task_type", "prompt", "seed_metadata"]]


def _assign_difficulty(df: pd.DataFrame, default: int = 2) -> pd.Series:
    template_map = {
        "hypothesis": 3,
        "methodology": 3,
        "explain": 2,
        "compare": 2,
        "analyze": 2,
        "short_answer": 1,
    }
    difficulty = df["task_type"].str.lower().map(template_map).fillna(default)
    return difficulty.astype(int)


def _stratified_sample(df: pd.DataFrame, total: int, per_domain: int) -> pd.DataFrame:
    if total <= 0:
        raise ValueError("Total sample size must be positive.")
    if per_domain <= 0:
        raise ValueError("Per-domain sample size must be positive.")

    selected_frames: List[pd.DataFrame] = []
    for domain, subset in df.groupby("domain"):
        take = min(per_domain, len(subset))
        selected_frames.append(subset.sample(n=take, random_state=42))
    combined = pd.concat(selected_frames, ignore_index=True)
    if len(combined) > total:
        combined = combined.sample(n=total, random_state=42)
    return combined.reset_index(drop=True)


def prepare_prompts(
    source: Path,
    output: Path,
    total: int = DEFAULT_TOTAL,
    per_domain: int = DEFAULT_PER_DOMAIN,
) -> pd.DataFrame:
    files = _collect_source_files(source)
    raw = _load_inputs(files)
    raw = raw.drop_duplicates(subset=["user_prompt"])
    normalised = _normalise_dataframe(raw)
    sampled = _stratified_sample(normalised, total=total, per_domain=per_domain)
    sampled["difficulty"] = _assign_difficulty(sampled)
    sampled.insert(0, "id", range(1, len(sampled) + 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output, index=False)

    # Store metadata alongside the parquet for reproducibility
    metadata = {
        "source_files": [str(path) for path in files],
        "total_requested": total,
        "per_domain_requested": per_domain,
        "records_written": len(sampled),
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare evaluation prompt parquet.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Directory or parquet file containing teacher inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination parquet path for generated prompts.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=DEFAULT_TOTAL,
        help="Total number of prompts to sample.",
    )
    parser.add_argument(
        "--per-domain",
        type=int,
        default=DEFAULT_PER_DOMAIN,
        help="Number of prompts to sample per domain before global cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_prompts(
        source=args.source,
        output=args.output,
        total=args.total,
        per_domain=args.per_domain,
    )
    print(f"[prepare-prompts] Wrote {len(result)} prompts to {args.output}")


if __name__ == "__main__":
    main()

