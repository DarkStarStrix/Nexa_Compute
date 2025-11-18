"""Run LLM-as-judge scoring for generated evaluation outputs."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from nexa_eval.clients import OpenRouterClient, OpenRouterConfig, OpenRouterRequest

DEFAULT_OUTPUT_DIR = Path("data/processed/evaluation/outputs")
DEFAULT_JUDGMENT_DIR = Path("data/processed/evaluation/judgments")
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_SYSTEM_PROMPT = """You are an expert scientific reviewer. You will be given:

- The domain and task type
- The original user prompt
- The model's answer

Score the answer from 1 (poor) to 5 (excellent) on these dimensions:
1. Scientific Correctness
2. Methodological Coherence
3. Specificity & Practicality
4. Clarity & Structure
5. Risk of Hallucination (5 = low hallucination risk)

Return a JSON object:
{
  "correctness": 1-5,
  "methodology": 1-5,
  "specificity": 1-5,
  "clarity": 1-5,
  "hallucination_safety": 1-5,
  "comments": "short explanation"
}

Do not include any additional text."""


@dataclass
class JudgeSpec:
    """Configuration describing the judge model."""

    model_id: str = DEFAULT_JUDGE_MODEL
    temperature: float = 0.0
    max_tokens: int = 512


def load_outputs(directory: Path) -> Dict[str, Path]:
    """Map model_id to parquet path from the output directory."""

    mapping: Dict[str, Path] = {}
    for path in directory.glob("outputs_*.parquet"):
        model_part = path.stem.replace("outputs_", "")
        model_id = model_part.replace("_", "/")
        mapping[model_id] = path
    return mapping


def build_user_prompt(record: pd.Series) -> str:
    """Construct the judge prompt from a record."""

    return (
        f"Domain: {record.get('domain', 'unknown')}\n"
        f"Task Type: {record.get('task_type', 'unknown')}\n\n"
        f"Original Prompt:\n{record.get('prompt', '')}\n\n"
        f"Model Answer:\n{record.get('output', '')}\n"
    )


def parse_judge_response(payload: str) -> Optional[Dict[str, object]]:
    """Parse JSON from judge response."""

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    required = [
        "correctness",
        "methodology",
        "specificity",
        "clarity",
        "hallucination_safety",
        "comments",
    ]
    if not all(key in parsed for key in required):
        return None
    try:
        return {
            "correctness": int(parsed["correctness"]),
            "methodology": int(parsed["methodology"]),
            "specificity": int(parsed["specificity"]),
            "clarity": int(parsed["clarity"]),
            "hallucination_safety": int(parsed["hallucination_safety"]),
            "comments": str(parsed["comments"]),
        }
    except (TypeError, ValueError):
        return None


def judge_model_outputs(
    outputs_df: pd.DataFrame,
    *,
    judge: JudgeSpec,
    client_config: Optional[OpenRouterConfig],
    dry_run: bool = False,
    label: str = "",
    max_workers: int = 1,
) -> pd.DataFrame:
    """Produce judgment scores for all rows in the outputs dataframe."""

    records: List[Dict[str, object]] = []
    rows = outputs_df.to_dict("records")
    progress = tqdm(
        total=len(rows),
        desc=f"Judging {label or 'outputs'}",
        unit="qa",
        leave=False,
    )

    if dry_run:
        for row in rows:
            records.append(
                {
                    "id": row["id"],
                    "model_id": row["model_id"],
                    "correctness": 3,
                    "methodology": 3,
                    "specificity": 3,
                    "clarity": 3,
                    "hallucination_safety": 3,
                    "comments": "Dry-run placeholder score.",
                }
            )
            progress.update(1)
        progress.close()
        return pd.DataFrame(records)

    if client_config is None:
        raise RuntimeError("OpenRouterConfig instance required when dry_run is False")

    max_workers = max(1, max_workers)

    def worker(row: Dict[str, object]) -> Dict[str, object]:
        prompt_text = build_user_prompt(row)
        request = OpenRouterRequest(
            prompt=prompt_text,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
            metadata={"prompt_id": int(row["id"])},
        )
        attempts = 0
        backoff = client_config.retry_backoff
        while True:
            client = OpenRouterClient(client_config)
            try:
                response = client.generate([request], model=judge.model_id, batch_size=1)[0]
                parsed = parse_judge_response(response.output_text)
                if parsed is None:
                    parsed = {
                        "correctness": 0,
                        "methodology": 0,
                        "specificity": 0,
                        "clarity": 0,
                        "hallucination_safety": 0,
                        "comments": "Unable to parse judge response.",
                    }
                return {
                    "id": row["id"],
                    "model_id": row["model_id"],
                    **parsed,
                }
            except Exception:
                attempts += 1
                if attempts > client_config.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= client_config.retry_backoff
            finally:
                client.close()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, row) for row in rows]
        try:
            for future in as_completed(futures):
                records.append(future.result())
                progress.update(1)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise

    progress.close()
    records.sort(key=lambda item: item.get("id", 0))
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge scoring.")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory with outputs_{model}.parquet files.")
    parser.add_argument("--judgment-dir", type=Path, default=DEFAULT_JUDGMENT_DIR, help="Directory to write judgments.")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model identifier for OpenRouter.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls and emit placeholder scores.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of rows per model for testing.")
    parser.add_argument("--max-workers", type=int, default=12, help="Max concurrent workers for OpenRouter judge calls.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional list of model IDs to judge (e.g. 'google/gemini-2.5-pro').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_outputs(args.outputs_dir)
    args.judgment_dir.mkdir(parents=True, exist_ok=True)

    judge = JudgeSpec(
        model_id=args.judge_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    target_models = set(args.models) if args.models else None
    client_config: Optional[OpenRouterConfig] = None
    if not args.dry_run:
        client_config = OpenRouterConfig(model=judge.model_id)

    processed_any = False
    for model_id, path in mapping.items():
        if target_models and model_id not in target_models:
            continue

        print(f"[judge] Processing {model_id} from {path}")
        outputs_df = pd.read_parquet(path)
        if args.limit:
            outputs_df = outputs_df.head(args.limit)

        df = judge_model_outputs(
            outputs_df,
            judge=judge,
            client_config=client_config,
            dry_run=args.dry_run,
            label=model_id,
            max_workers=args.max_workers,
        )

        output_path = args.judgment_dir / f"judgments_{model_id.replace('/', '_')}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"[judge] Saved {len(df)} judgments to {output_path}")
        processed_any = True

    if target_models and not processed_any:
        print("[judge] No matching models found for the provided --models filter.")


if __name__ == "__main__":
    main()

