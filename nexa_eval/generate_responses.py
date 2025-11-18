"""Generate evaluation outputs for a set of models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from nexa_eval.clients import (
    LocalAPIConfig,
    LocalGenerationRequest,
    LocalInferenceClient,
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterRequest,
)

DEFAULT_PROMPTS = Path("data/processed/evaluation/prompts/prompts.parquet")
DEFAULT_OUTPUT_DIR = Path("data/processed/evaluation/outputs")
DEFAULT_SYSTEM_PROMPT = (
    "You are a rigorous scientific assistant. For each user task, you must:\n"
    "1. Explicitly state the hypothesis or question.\n"
    "2. Provide a step-by-step methodology suitable for a real lab or simulation environment.\n"
    "3. Clearly list assumptions and potential failure modes.\n"
    "4. Be concise but technically precise. Do NOT invent fake references."
)

DEFAULT_MODELS = [
    {"id": "NexaSci-Falcon", "type": "local", "parameters": {"base_url": "http://127.0.0.1:8000"}},
    {"id": "google/gemini-2.5-pro", "type": "openrouter"},
    {"id": "openai/gpt-5", "type": "openrouter"},
    {"id": "openai/gpt-4o-mini", "type": "openrouter"},
    {"id": "anthropic/claude-sonnet-4.5", "type": "openrouter"},
]


@dataclass
class ModelSpec:
    """Description of a model participating in the generation sweep."""

    id: str
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


def load_model_specs(path: Optional[Path]) -> List[ModelSpec]:
    """Load model definitions from JSON; fall back to defaults."""

    if path is None:
        return [ModelSpec(**item) for item in DEFAULT_MODELS]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ModelSpec(**item) for item in payload]


def iter_prompts(df: pd.DataFrame, *, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield prompt records up to the optional limit."""

    for index, row in df.iterrows():
        if limit is not None and index >= limit:
            break
        yield {
            "id": int(row["id"]),
            "domain": row.get("domain"),
            "task_type": row.get("task_type"),
            "prompt": row.get("prompt"),
            "seed_metadata": row.get("seed_metadata"),
        }


def generate_for_local(
    spec: ModelSpec,
    prompts: Iterable[Dict[str, Any]],
    *,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Generate responses using the local FastAPI inference server."""

    config = LocalAPIConfig(
        base_url=spec.parameters.get("base_url", "http://127.0.0.1:8000"),
        infer_endpoint=spec.parameters.get("infer_endpoint", "/infer"),
        timeout_s=int(spec.parameters.get("timeout_s", 60)),
        default_temperature=float(spec.parameters.get("temperature", temperature)),
        default_max_tokens=int(spec.parameters.get("max_tokens", max_tokens)),
    )
    client = LocalInferenceClient(config)
    results: List[Dict[str, Any]] = []
    try:
        requests_ = [
            LocalGenerationRequest(
                prompt=item["prompt"],
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                metadata={"prompt_id": item["id"]},
            )
            for item in prompts
        ]
        progress = tqdm(
            total=len(requests_),
            desc=f"{spec.id} (local)",
            unit="req",
            leave=False,
        )
        for start in range(0, len(requests_), batch_size):
            batch = requests_[start : start + batch_size]
            responses = client.generate(batch, batch_size=batch_size)
            for request_item, response in zip(batch, responses):
                results.append(
                    {
                        "id": request_item.metadata["prompt_id"],
                        "model_id": spec.id,
                        "output": response.output_text,
                        "tokens_in": 0,
                        "tokens_out": response.tokens,
                        "latency_ms": response.latency_ms,
                        "raw_response": response.raw,
                    }
                )
                progress.update(1)
        progress.close()
    finally:
        client.close()
    return results


def generate_for_openrouter(
    spec: ModelSpec,
    prompts: Iterable[Dict[str, Any]],
    *,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    batch_size: int,
    max_workers: int,
) -> List[Dict[str, Any]]:
    """Generate responses using the OpenRouter API with concurrency."""

    config = OpenRouterConfig(
        model=spec.id,
        api_key=spec.parameters.get("api_key"),
        base_url=spec.parameters.get("base_url", "https://openrouter.ai/api/v1"),
        timeout_s=int(spec.parameters.get("timeout_s", 60)),
        default_temperature=float(spec.parameters.get("temperature", temperature)),
        default_max_tokens=int(spec.parameters.get("max_tokens", max_tokens)),
    )

    requests_ = [
        OpenRouterRequest(
            prompt=item["prompt"],
            system_prompt=system_prompt,
            metadata={"prompt_id": item["id"]},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for item in prompts
    ]

    requested_workers = spec.parameters.get("max_workers")
    if requested_workers is not None:
        max_workers = int(requested_workers)
    if max_workers <= 0:
        max_workers = 1

    results: List[Dict[str, Any]] = []
    progress = tqdm(
        total=len(requests_),
        desc=f"{spec.id} (OpenRouter)",
        unit="req",
        leave=False,
    )

    def worker(request: OpenRouterRequest) -> Dict[str, Any]:
        backoff = config.retry_backoff
        attempts = 0
        while True:
            client = OpenRouterClient(config)
            try:
                response = client.generate([request], model=spec.id, batch_size=1)[0]
                prompt_id = response.metadata.get("prompt_id") if response.metadata else None
                
                if not response.output_text and response.usage.completion_tokens > 0:
                    import warnings
                    warnings.warn(
                        f"Empty output from {spec.id} but {response.usage.completion_tokens} tokens consumed. "
                        f"Finish reason: {response.raw.get('choices', [{}])[0].get('finish_reason', 'unknown') if response.raw else 'unknown'}"
                    )
                
                return {
                    "id": int(prompt_id) if prompt_id is not None else None,
                    "model_id": spec.id,
                    "output": response.output_text or "",
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens,
                    "latency_ms": response.latency_ms,
                    "raw_response": response.raw,
                }
            except Exception as exc:
                attempts += 1
                if attempts > config.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= config.retry_backoff
            finally:
                client.close()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, request) for request in requests_]
        try:
            for future in as_completed(futures):
                results.append(future.result())
                progress.update(1)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    progress.close()
    return results


def build_records(
    prompts_df: pd.DataFrame,
    model_results: List[Dict[str, Any]],
    system_prompt: str,
) -> pd.DataFrame:
    """Combine model outputs with the original prompt metadata."""

    joined = pd.DataFrame(model_results)
    merged = prompts_df.merge(joined, on="id", how="inner")
    merged["system_prompt"] = system_prompt
    columns = [
        "id",
        "model_id",
        "domain",
        "task_type",
        "prompt",
        "system_prompt",
        "output",
        "tokens_in",
        "tokens_out",
        "latency_ms",
    ]
    if "raw_response" in merged.columns:
        columns.append("raw_response")
    return merged[columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation outputs.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS, help="Path to prompts parquet.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store outputs.")
    parser.add_argument("--models-config", type=Path, default=None, help="Optional JSON list of model specs.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens for model responses (default: 2048).")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for prompts (debug).")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without invoking models.")
    parser.add_argument(
        "--openrouter-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent OpenRouter requests (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts_df = pd.read_parquet(args.prompts)
    prompts_iter = list(iter_prompts(prompts_df, limit=args.limit))
    model_specs = load_model_specs(args.models_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for spec in model_specs:
        print(f"[generate] Processing model {spec.id} ({spec.type}) with {len(prompts_iter)} prompts")
        start = time.perf_counter()

        if args.dry_run:
            print("[generate] Dry run: skipping model invocation")
            continue

        model_max_tokens = spec.parameters.get("max_tokens", args.max_tokens) if spec.parameters else args.max_tokens
        if isinstance(model_max_tokens, str):
            model_max_tokens = int(model_max_tokens)
        
        if spec.type == "local":
            results = generate_for_local(
                spec,
                prompts_iter,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_tokens=model_max_tokens,
                temperature=args.temperature,
                batch_size=args.batch_size,
            )
        elif spec.type == "openrouter":
            results = generate_for_openrouter(
                spec,
                prompts_iter,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_tokens=model_max_tokens,
                temperature=args.temperature,
                batch_size=args.batch_size,
                max_workers=args.openrouter_workers,
            )
        else:
            print(f"[generate] Unsupported model type {spec.type}", file=sys.stderr)
            continue

        elapsed = time.perf_counter() - start
        df = build_records(prompts_df, results, DEFAULT_SYSTEM_PROMPT)
        output_path = args.output_dir / f"outputs_{spec.id.replace('/', '_')}.parquet"
        df.to_parquet(output_path, index=False)
        metrics_path = output_path.with_suffix(".json")
        metrics = {
            "model_id": spec.id,
            "num_records": len(df),
            "elapsed_seconds": elapsed,
            "avg_latency_ms": float(df["latency_ms"].mean()) if len(df) else 0.0,
            "total_prompt_tokens": int(df["tokens_in"].sum()) if len(df) else 0,
            "total_completion_tokens": int(df["tokens_out"].sum()) if len(df) else 0,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[generate] Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()

