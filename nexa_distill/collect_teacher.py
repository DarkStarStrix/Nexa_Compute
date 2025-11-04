"""Teacher collection script for the Nexa Distill pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional

import pandas as pd
import yaml
from tqdm import tqdm

from . import CONFIGS_DIR, PROMPTS_DIR
from .utils import (
    OpenAIClient,
    PromptRequest,
    PromptResult,
    get_logger,
    normalize_whitespace,
    read_parquet,
    write_parquet,
)


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for teacher collection."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Input parquet with prompts")
    parser.add_argument("--dst", type=Path, required=True, help="Destination parquet path")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "distill_config.yaml")
    parser.add_argument("--teacher", type=str, default="gpt-4o-mini", help="Model identifier")
    parser.add_argument("--system-prompt", type=Path, default=None)
    parser.add_argument("--prompt-column", type=str, default="prompt_text")
    parser.add_argument("--context-column", type=str, default="context")
    parser.add_argument("--task-type-column", type=str, default="task_type")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, emit stub outputs")
    parser.add_argument("--api-key", type=str, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    """Load YAML configuration from ``path``."""

    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_prompt_templates() -> Dict[str, str]:
    """Load default hypothesis and methodology prompt templates."""

    templates: Dict[str, str] = {}
    for task_name in ("hypothesis", "methodology"):
        template_path = PROMPTS_DIR / f"{task_name}.txt"
        if template_path.exists():
            templates[task_name] = template_path.read_text(encoding="utf-8").strip()
    return templates


def resolve_prompt(
    row: Mapping[str, object],
    *,
    prompt_column: str,
    context_column: str,
    task_type_column: str,
    templates: Mapping[str, str],
) -> str:
    """Build the teacher prompt for a single record."""

    prompt_value = str(row.get(prompt_column, ""))
    if prompt_value:
        return normalize_whitespace(prompt_value)

    context = normalize_whitespace(str(row.get(context_column, "")))
    task_type = str(row.get(task_type_column, "hypothesis"))
    template = templates.get(task_type.lower())
    if not template:
        raise KeyError(
            f"Prompt template for task '{task_type}' is missing. Provide a prompt column "
            "or ensure templates exist."
        )
    return template.format(context=context)


def build_requests(
    df: pd.DataFrame,
    *,
    prompt_column: str,
    context_column: str,
    task_type_column: str,
    system_prompt: Optional[str],
) -> List[PromptRequest]:
    """Convert DataFrame rows into prompt requests."""

    templates = load_prompt_templates()
    requests: List[PromptRequest] = []
    for _, row in df.iterrows():
        prompt = resolve_prompt(
            row,
            prompt_column=prompt_column,
            context_column=context_column,
            task_type_column=task_type_column,
            templates=templates,
        )
        metadata = {
            "task_type": str(row.get(task_type_column, "")),
        }
        requests.append(PromptRequest(user_prompt=prompt, system_prompt=system_prompt, metadata=metadata))
    return requests


def apply_results(
    df: pd.DataFrame,
    results: List[PromptResult],
) -> pd.DataFrame:
    """Attach model outputs to the original DataFrame."""

    outputs = df.copy().reset_index(drop=True)
    outputs["teacher_output"] = [result.output_text for result in results]
    outputs["model_id"] = [result.model_id for result in results]
    outputs["latency_ms"] = [result.latency_ms for result in results]
    outputs["prompt_tokens"] = [result.usage.get("prompt_tokens", 0) for result in results]
    outputs["completion_tokens"] = [result.usage.get("completion_tokens", 0) for result in results]
    outputs["total_tokens"] = [result.usage.get("total_tokens", 0) for result in results]
    return outputs


def run_collection(args: argparse.Namespace) -> None:
    """Execute the teacher collection workflow."""

    config = load_yaml(args.config)
    LOGGER.info("Loaded config", extra={"config_path": str(args.config)})

    df = read_parquet(args.src)
    if args.max_samples:
        df = df.head(args.max_samples)
    LOGGER.info("Loaded dataset", extra={"rows": len(df), "path": str(args.src)})

    system_prompt_text = None
    if args.system_prompt:
        system_prompt_text = args.system_prompt.read_text(encoding="utf-8").strip()

    requests = build_requests(
        df,
        prompt_column=args.prompt_column,
        context_column=args.context_column,
        task_type_column=args.task_type_column,
        system_prompt=system_prompt_text,
    )
    LOGGER.info("Prepared prompt requests", extra={"count": len(requests)})

    if args.dry_run:
        LOGGER.warning("Dry run enabled — writing placeholder outputs only")
        mocked_results = [
            PromptResult(
                output_text="[DRY RUN] Teacher output not generated.",
                latency_ms=0,
                model_id=args.teacher,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            for _ in requests
        ]
        enriched_df = apply_results(df, mocked_results)
        write_parquet(enriched_df, args.dst)
        LOGGER.info("Dry run output written", extra={"dst": str(args.dst)})
        return

    client = OpenAIClient(api_key=args.api_key, default_model=args.teacher)
    batched_results: List[PromptResult] = []
    for start_idx in tqdm(range(0, len(requests), args.batch_size), desc="Collecting"):
        batch = requests[start_idx : start_idx + args.batch_size]
        batch_results = client.generate_batch(batch)
        batched_results.extend(batch_results)

    enriched_df = apply_results(df, batched_results)
    write_parquet(enriched_df, args.dst)
    LOGGER.info(
        "Teacher outputs written",
        extra={"rows": len(enriched_df), "dst": str(args.dst), "model": args.teacher},
    )


def main() -> None:
    """CLI entrypoint."""

    run_collection(parse_args())


if __name__ == "__main__":  # pragma: no cover - executable script
    main()

