"""Regenerate rejected samples using a stricter teacher prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .collect_teacher import apply_results, build_requests
from .utils import OpenAIClient, PromptResult, get_logger, read_jsonl, read_parquet, write_parquet


LOGGER = get_logger(__name__)
TARGET_ACTIONS = {"reject", "regenerate"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Filtered parquet dataset")
    parser.add_argument("--annotations", type=Path, required=True, help="JSONL labels from inspector")
    parser.add_argument("--dst", type=Path, required=True, help="Destination parquet path")
    parser.add_argument("--teacher", type=str, default="gpt-4o-mini")
    parser.add_argument("--system-prompt", type=Path, default=None)
    parser.add_argument("--prompt-column", type=str, default="prompt_text")
    parser.add_argument("--context-column", type=str, default="context")
    parser.add_argument("--task-type-column", type=str, default="task_type")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_annotations(path: Path) -> List[dict]:
    """Load annotations and filter for rows requiring regeneration."""

    annotations = [record for record in read_jsonl(path) if record.get("action") in TARGET_ACTIONS]
    LOGGER.info(
        "Loaded annotations",
        extra={"total": len(annotations), "path": str(path), "actions": list(TARGET_ACTIONS)},
    )
    return annotations


def subset_rows(df: pd.DataFrame, annotations: List[dict]) -> pd.DataFrame:
    """Return subset of ``df`` whose indices match annotation entries."""

    indices = [int(record["index"]) for record in annotations]
    subset = df.loc[df.index.intersection(indices)].copy()
    mapping = {int(record["index"]): record["action"] for record in annotations}
    subset["annotation_action"] = subset.index.map(mapping.get)
    return subset


def run_regeneration(args: argparse.Namespace) -> None:
    """Execute regeneration workflow."""

    df = read_parquet(args.dataset)
    annotations = load_annotations(args.annotations)
    if not annotations:
        LOGGER.warning("No annotations requiring regeneration were found. Exiting.")
        return

    subset = subset_rows(df, annotations)
    if subset.empty:
        LOGGER.warning("No matching rows found in dataset for provided annotations.")
        return

    system_prompt_text = None
    if args.system_prompt:
        system_prompt_text = args.system_prompt.read_text(encoding="utf-8").strip()

    requests = build_requests(
        subset,
        prompt_column=args.prompt_column,
        context_column=args.context_column,
        task_type_column=args.task_type_column,
        system_prompt=system_prompt_text,
    )

    if args.dry_run:
        LOGGER.warning("Dry run enabled — skipping regeneration API calls")
        mocked_results = [
            PromptResult(
                output_text="[DRY RUN] Regenerated output not produced.",
                latency_ms=0,
                model_id=args.teacher,
                usage={},
            )
            for _ in requests
        ]
        enriched = apply_results(subset, mocked_results)
        write_parquet(enriched, args.dst)
        LOGGER.info("Dry run regeneration parquet written", extra={"dst": str(args.dst)})
        return

    client = OpenAIClient(api_key=args.api_key, default_model=args.teacher)
    results = client.generate_batch(requests)
    enriched = apply_results(subset, results)
    enriched["regenerated"] = True
    write_parquet(enriched, args.dst)
    LOGGER.info(
        "Regenerated samples written",
        extra={"rows": len(enriched), "dst": str(args.dst), "model": args.teacher},
    )


def main() -> None:
    """CLI entrypoint."""

    run_regeneration(parse_args())


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()

