"""Convert filtered distillation outputs into SFT-ready JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .utils import get_logger, normalize_whitespace, read_parquet, write_jsonl, write_parquet


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Filtered parquet path")
    parser.add_argument("--regen", type=Path, default=None, help="Optional regenerated parquet")
    parser.add_argument("--dst-jsonl", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--dst-parquet", type=Path, default=None, help="Optional parquet copy")
    parser.add_argument("--prompt-column", type=str, default="prompt_text")
    parser.add_argument("--context-column", type=str, default="context")
    parser.add_argument("--task-type-column", type=str, default="task_type")
    parser.add_argument("--output-column", type=str, default="teacher_output")
    return parser.parse_args()


def merge_regenerated(base: pd.DataFrame, regen_path: Optional[Path]) -> pd.DataFrame:
    """Merge regenerated rows into the base dataset."""

    if regen_path is None:
        return base
    regenerated = read_parquet(regen_path)
    LOGGER.info("Loaded regenerated samples", extra={"rows": len(regenerated), "path": str(regen_path)})
    merged = base.copy()
    for idx, row in regenerated.iterrows():
        merged.loc[idx, :] = row
    merged["regenerated"] = merged.get("regenerated", False) | merged.index.isin(regenerated.index)
    return merged


def build_input_text(
    row: pd.Series,
    *,
    prompt_column: str,
    context_column: str,
    task_type_column: str,
) -> str:
    """Compose the input text for SFT."""

    prompt = normalize_whitespace(str(row.get(prompt_column, "")))
    if prompt:
        return prompt

    context = normalize_whitespace(str(row.get(context_column, "")))
    task_type = row.get(task_type_column, "").upper()
    return f"[TASK: {task_type}]\n{context}" if context else context


def build_record(row: pd.Series, *, output_column: str, prompt_column: str, context_column: str, task_type_column: str) -> Dict:
    """Build a JSONL record for the SFT dataset."""

    input_text = build_input_text(
        row,
        prompt_column=prompt_column,
        context_column=context_column,
        task_type_column=task_type_column,
    )
    output_text = normalize_whitespace(str(row.get(output_column, "")))
    meta = {
        "task_type": row.get(task_type_column),
        "model_id": row.get("model_id"),
        "regenerated": bool(row.get("regenerated", False)),
        "source_index": int(getattr(row, "name", -1)),
    }
    return {"input": input_text, "output": output_text, "meta": meta}


def run_packaging(args: argparse.Namespace) -> None:
    """Execute conversion to SFT format."""

    df = read_parquet(args.src)
    LOGGER.info("Loaded filtered dataset", extra={"rows": len(df), "path": str(args.src)})
    merged = merge_regenerated(df, args.regen)

    records = [
        build_record(
            row,
            output_column=args.output_column,
            prompt_column=args.prompt_column,
            context_column=args.context_column,
            task_type_column=args.task_type_column,
        )
        for _, row in merged.iterrows()
        if row.get(args.output_column)
    ]

    write_jsonl(records, args.dst_jsonl)
    LOGGER.info(
        "SFT JSONL written",
        extra={"records": len(records), "dst": str(args.dst_jsonl)},
    )

    if args.dst_parquet:
        output_df = pd.DataFrame(records)
        write_parquet(output_df, args.dst_parquet)
        LOGGER.info("SFT parquet copy written", extra={"dst": str(args.dst_parquet)})


def main() -> None:
    """CLI entrypoint."""

    run_packaging(parse_args())


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

