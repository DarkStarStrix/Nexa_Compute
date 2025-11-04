"""Filter teacher outputs using heuristic quality gates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import yaml

from . import CONFIGS_DIR
from .utils import FilterConfig, basic_filters, get_logger, read_parquet, write_parquet


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Input parquet path")
    parser.add_argument("--dst", type=Path, required=True, help="Filtered parquet destination")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "filters.yaml")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def load_filter_config(path: Path) -> FilterConfig:
    """Load filter configuration from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    verbs: Iterable[str] | None = raw.get("allowed_action_verbs")
    if verbs is not None:
        verbs = tuple(str(verb).strip() for verb in verbs if str(verb).strip())

    return FilterConfig(
        min_char_length=int(raw.get("min_char_length", FilterConfig.min_char_length)),
        min_token_length=int(raw.get("min_token_length", FilterConfig.min_token_length)),
        require_action_verb=bool(raw.get("require_action_verb", True)),
        ban_citations=bool(raw.get("ban_citations", True)),
        action_verbs=verbs,
    )


def apply_filters(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """Apply heuristic filters to the DataFrame."""

    results, passes = [], []
    for _, row in df.iterrows():
        flags = basic_filters(row.to_dict(), config=config)
        passes.append(flags["passes_all"])
        results.append(flags)

    flags_df = pd.DataFrame(results)
    combined = pd.concat([df.reset_index(drop=True), flags_df], axis=1)
    combined["quality_pass"] = passes
    return combined


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary statistics for the filtering stage."""

    total = len(df)
    passed = int(df["quality_pass"].sum())
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_completion_tokens": float(df["completion_tokens"].mean()) if "completion_tokens" in df else 0.0,
    }


def run_filtering(args: argparse.Namespace) -> None:
    """Execute filtering workflow."""

    config = load_filter_config(args.config)
    df = read_parquet(args.src)
    LOGGER.info("Loaded teacher dataset", extra={"rows": len(df), "path": str(args.src)})

    filtered = apply_filters(df, config)
    write_parquet(filtered[filtered["quality_pass"]], args.dst)
    LOGGER.info(
        "Filtered dataset written",
        extra={"rows": int(filtered["quality_pass"].sum()), "dst": str(args.dst)},
    )

    report = summarize(filtered)
    LOGGER.info("Filter summary", extra=report)
    if args.report:
        args.report.write_text(yaml.safe_dump(report), encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""

    run_filtering(parse_args())


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

