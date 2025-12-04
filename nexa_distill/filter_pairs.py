"""Filter teacher outputs using heuristic quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import yaml

from . import CONFIGS_DIR
from .utils import FilterConfig as LegacyFilterConfig, basic_filters, get_logger, read_parquet, write_parquet

# Try to import Rust integration
try:
    from nexa_compute.data.rust_quality import filter_corpus, FilterConfig as RustFilterConfig
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Input parquet path")
    parser.add_argument("--dst", type=Path, required=True, help="Filtered parquet destination")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "filters.yaml")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--use-rust", action="store_true", help="Force use of Rust engine")
    return parser.parse_args()


def load_filter_config(path: Path) -> LegacyFilterConfig:
    """Load filter configuration from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    verbs: Iterable[str] | None = raw.get("allowed_action_verbs")
    if verbs is not None:
        verbs = tuple(str(verb).strip() for verb in verbs if str(verb).strip())

    return LegacyFilterConfig(
        min_char_length=int(raw.get("min_char_length", LegacyFilterConfig.min_char_length)),
        min_token_length=int(raw.get("min_token_length", LegacyFilterConfig.min_token_length)),
        require_action_verb=bool(raw.get("require_action_verb", True)),
        ban_citations=bool(raw.get("ban_citations", True)),
        action_verbs=verbs,
    )


def apply_filters(df: pd.DataFrame, config: LegacyFilterConfig) -> pd.DataFrame:
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

    # Prefer Rust if available and not explicitly disabled (though explicit flag not added to args in pipeline call yet)
    # args might not have use_rust if called from pipeline.py without update.
    use_rust = getattr(args, "use_rust", False) or RUST_AVAILABLE

    if use_rust and RUST_AVAILABLE:
        LOGGER.info("Using Rust-powered filtering engine")
        # Map config
        # We need to parse the config YAML manually to map to Rust config or use LegacyFilterConfig values
        # LegacyFilterConfig is a dataclass (from utils).
        legacy_config = load_filter_config(args.config)
        
        # Infer rejected path
        rejected_path = args.dst.parent / f"rejected_{args.dst.name}"
        
        rust_config = RustFilterConfig(
            text_column="teacher_output", # Assumption: filtering teacher output
            min_length=legacy_config.min_char_length,
            # bad_patterns mapping: currently Legacy has ban_citations logic hardcoded in basic_filters.
            # We would need to replicate that pattern list.
            # For now, we'll pass empty patterns unless we load them.
            dedup_enabled=True # Enable dedup by default in V4?
        )
        
        try:
            stats = filter_corpus(
                args.src,
                args.dst,
                rejected_path,
                rust_config
            )
            
            report = {
                "total": stats["total"],
                "passed": stats["kept"],
                "pass_rate": round(stats["kept"] / stats["total"], 4) if stats["total"] else 0.0,
                "engine": "rust"
            }
            LOGGER.info("Filter summary (Rust)", extra=report)
            if args.report:
                args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return
        except Exception as e:
            LOGGER.warning(f"Rust filtering failed, falling back to Python: {e}")

    # Fallback to Python
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
