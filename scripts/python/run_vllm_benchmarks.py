#!/usr/bin/env python3
"""
Benchmark vLLM inference configurations for throughput and latency.

Usage:
    python vllm_bench.py --model-id tiiuae/falcon-10b-instruct --output-dir results/v1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VENDOR = ROOT / "vendor"
for candidate in (ROOT, SRC, VENDOR):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

load_dotenv(ROOT / ".env")

# Local imports
from nexa_eval.generate import (
    VLLMGenerationConfig,
    run_vllm_benchmark,
    save_benchmark_artifacts,
)
from nexa_eval.judge import Rubric, judge_metrics


DEFAULT_PROMPTS: List[str] = [
    "Explain how superconductors enable near-lossless power transmission in urban grids.",
    "Design a reproducible lab experiment to measure photosynthesis rate in freshwater algae cultures.",
    "Outline the reaction mechanism for catalytic hydrogenation of carbon dioxide into methanol.",
    "Compare the advantages and limitations of CRISPR base editing versus prime editing for point mutation repair.",
    "Describe the process of nucleation in cloud seeding and its impact on precipitation rates.",
    "Given a perovskite solar cell stack, propose diagnostic tests to isolate rapid efficiency degradation.",
    "What experimental controls are required to validate a new mRNA vaccine delivery nanoparticle?",
    "How does quantum confinement influence the band gap of semiconductor nanocrystals?",
    "Provide a method to quantify microplastic concentration in coastal water samples.",
    "Explain the role of spin-orbit coupling in topological insulators.",
    "Detail the steps for deriving the Navier-Stokes equations from first principles.",
    "Propose a protocol for validating a computational protein design using biophysical assays.",
]

# Throughput targets / SLOs
DEFAULT_RUBRICS = [
    Rubric(metric="throughput_tokens_per_s", threshold=1000.0, direction="max"),
    Rubric(metric="avg_latency_s", threshold=2.5, direction="min"),
]


def _build_configs(model_id: str, prompt_path: Path | None) -> List[VLLMGenerationConfig]:
    """Return memory-stable configs for the benchmark run."""
    prompts = None if prompt_path else DEFAULT_PROMPTS

    return [
        VLLMGenerationConfig(
            name="tp1_len2048_batch4_long",
            model_id=model_id,
            prompt_path=prompt_path,
            prompts=prompts,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2048,
            max_new_tokens=1024,
            max_num_seqs=4,
            dtype="bfloat16",
            trust_remote_code=True,
        ),
    ]


def _write_summary(records: List[Dict[str, float]], out_dir: Path) -> Path:
    summary_df = pd.DataFrame(records)
    summary_path = out_dir / "summary.parquet"
    summary_df.to_parquet(summary_path, index=False)

    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM inference configurations.")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face model ID.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "vllm_benchmarks",
        help="Output directory for metrics & artifacts.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=None,
        help="Optional path to prompts (JSONL or .txt). Overrides built-in prompts.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = _build_configs(args.model_id, args.prompt_path)

    summary_records: List[Dict[str, float]] = []
    per_prompt_tables: List[pd.DataFrame] = []

    for cfg in configs:
        print(f"\n=== Benchmarking: {cfg.name} ===")
        per_prompt_df, metrics = run_vllm_benchmark(cfg)

        cfg_dir = args.output_dir / cfg.name
        save_benchmark_artifacts(per_prompt_df, metrics, cfg_dir)

        verdicts = judge_metrics(metrics, DEFAULT_RUBRICS)
        metrics.update({f"pass_{k}": v for k, v in verdicts.items()})

        summary_records.append(metrics)
        metric_overview = {
            key: value
            for key, value in metrics.items()
            if key
            in {
                "config_name",
                "throughput_tokens_per_s",
                "avg_latency_s",
                "throughput_requests_per_s",
                "runtime_s",
                "tensor_parallel_size",
                "gpu_memory_utilization",
                "max_num_seqs",
                "max_num_batched_tokens",
                "enable_prefix_caching",
                "enable_chunked_prefill",
            }
            and not isinstance(value, (list, dict))
        }
        per_prompt_tables.append(per_prompt_df.assign(**metric_overview))

        print(
            f"✓ {cfg.name}: {metrics['throughput_tokens_per_s']:.1f} tok/s | "
            f"{metrics['avg_latency_s']:.3f} s avg latency"
        )

    summary_path = _write_summary(summary_records, args.output_dir)
    print(f"\n✅ Summary written to: {summary_path}")
    if per_prompt_tables:
        results_df = pd.concat(per_prompt_tables, ignore_index=True)
        results_path = args.output_dir / "results.parquet"
        results_df.to_parquet(results_path, index=False)
        print(f"Per-request results written to: {results_path}")


if __name__ == "__main__":
    main()
