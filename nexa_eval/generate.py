"""Generation utilities for evaluation tasks."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = ROOT / "vendor"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from nexa_compute.config import load_config  # type: ignore
from nexa_compute.data.pipeline import DataPipeline  # type: ignore
from nexa_compute.models.base import DEFAULT_MODEL_REGISTRY  # type: ignore
from nexa_compute.training.checkpoint import load_checkpoint  # type: ignore

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None


def generate_predictions(
    config_path: Path,
    checkpoint: Optional[Path] = None,
) -> Tuple[List[List[float]], List[int]]:
    """Run classification-style evaluation using the legacy trainer stack.

    Args:
        config_path: Path to the training configuration file.
        checkpoint: Optional checkpoint to restore weights from.

    Returns:
        Tuple of (probabilities, labels) for downstream metric computation.
    """
    cfg = load_config(config_path)
    pipeline = DataPipeline(cfg.data)
    dataloader = pipeline.dataloader(
        "validation", batch_size=cfg.evaluation.batch_size
    )
    model = DEFAULT_MODEL_REGISTRY.build(cfg.model)
    if checkpoint:
        state = load_checkpoint(checkpoint)
        model.load_state_dict(state["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    probs: List[List[float]] = []
    labels: List[int] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_probs = torch.softmax(outputs, dim=1).cpu().tolist()
            probs.extend(batch_probs)
            labels.extend(targets.tolist())
    return probs, labels


def _ensure_vllm_available() -> None:
    """Raise a descriptive error if vLLM is not installed."""
    if LLM is None or SamplingParams is None:
        raise RuntimeError(
            "vllm is not installed. Install it with `pip install vllm` "
            "inside the active environment to enable generation benchmarking."
        )


def _resolve_prompts_from_path(prompt_path: Path) -> List[str]:
    """Load prompts from a JSONL or plain-text file.

    Args:
        prompt_path: Input file containing prompts.

    Returns:
        List of non-empty prompt strings.
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt path {prompt_path} does not exist.")

    if prompt_path.suffix.lower() in {".json", ".jsonl"}:
        prompts: List[str] = []
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    candidate = payload.get("prompt") or payload.get("question")
                else:
                    candidate = payload
                if isinstance(candidate, str):
                    prompts.append(candidate.strip())
        return [item for item in prompts if item]

    # Treat everything else as plain-text, one prompt per line.
    with prompt_path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    return prompts


@dataclass
class VLLMGenerationConfig:
    """Configuration for running vLLM-based generation benchmarks."""

    name: str
    model_id: str
    prompts: Optional[Sequence[str]] = None
    prompt_path: Optional[Path] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[Sequence[str]] = None
    seed: Optional[int] = 42
    dtype: Optional[str] = "bfloat16"
    trust_remote_code: bool = True
    enforce_eager: bool = False
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    long_prefill_token_threshold: Optional[int] = None
    quantization: Optional[str] = None
    speculative_draft: Optional[str] = None
    replicas: Optional[int] = None
    device_affinity: Optional[str] = None

    def materialise_prompts(self) -> List[str]:
        """Return the list of prompts configured for the benchmark."""
        if self.prompts:
            expanded = [item.strip() for item in self.prompts if item and item.strip()]
            if not expanded:
                raise ValueError("Configured prompts list is empty after stripping.")
            return expanded
        if self.prompt_path:
            return _resolve_prompts_from_path(Path(self.prompt_path))
        raise ValueError(
            "No prompts supplied. Provide either `prompts` or `prompt_path`."
        )

    def sampling_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for `SamplingParams` construction."""
        params: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
        }
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.stop:
            params["stop"] = list(self.stop)
        return params

    def engine_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for the vLLM engine."""
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
        }
        if self.max_num_seqs is not None:
            kwargs["max_num_seqs"] = self.max_num_seqs
        if self.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.enable_prefix_caching:
            kwargs["enable_prefix_caching"] = True
        if self.enable_chunked_prefill:
            kwargs["enable_chunked_prefill"] = True
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization
        return kwargs

    def to_metadata(self) -> Dict[str, Any]:
        """Return serialisable metadata describing the config."""
        payload = asdict(self)
        if payload.get("prompts") is not None:
            payload["prompts"] = list(payload["prompts"])
        if payload.get("prompt_path") is not None:
            payload["prompt_path"] = str(payload["prompt_path"])
        payload["timestamp_utc"] = datetime.utcnow().isoformat()
        return payload


def run_vllm_benchmark(
    config: VLLMGenerationConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run generation with vLLM and capture per-request outputs and metrics.

    Args:
        config: Benchmark configuration describing engine and sampling settings.

    Returns:
        A tuple containing:
            - DataFrame with prompt/response level details.
            - Dictionary of aggregate metrics for the run.
    """
    _ensure_vllm_available()

    prompts = config.materialise_prompts()
    target_requests = config.max_num_seqs or len(prompts)
    if target_requests > 0 and len(prompts) < target_requests:
        repeats = math.ceil(target_requests / len(prompts))
        tiled = list(prompts) * repeats
        prompts = tiled[:target_requests]
    if not prompts:
        raise ValueError("No prompts available for benchmarking.")

    sampling_params = SamplingParams(**config.sampling_kwargs())
    llm = LLM(**config.engine_kwargs())

    request_outputs = None
    try:
        start_time = time.perf_counter()
        request_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        runtime_s = time.perf_counter() - start_time

        records: List[Dict[str, Any]] = []
        total_prompt_tokens = 0
        total_generated_tokens = 0

        for prompt, output in zip(prompts, request_outputs):
            prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
            completion = output.outputs[0] if output.outputs else None
            completion_text = completion.text if completion else ""
            completion_tokens = len(completion.token_ids) if completion else 0

            total_prompt_tokens += prompt_tokens
            total_generated_tokens += completion_tokens

            records.append(
                {
                    "config_name": config.name,
                    "prompt": prompt,
                    "response": completion_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            )

        df = pd.DataFrame(records)

        throughput_tps = (
            total_generated_tokens / runtime_s if runtime_s > 0 else float("nan")
        )
        requests_per_second = len(prompts) / runtime_s if runtime_s > 0 else float("nan")
        avg_latency_s = runtime_s / len(prompts) if prompts else float("nan")

        metrics: Dict[str, Any] = {
            "config_name": config.name,
            "model_id": config.model_id,
            "num_requests": len(prompts),
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_generated_tokens,
            "avg_output_tokens": (
                total_generated_tokens / len(prompts) if prompts else float("nan")
            ),
            "runtime_s": runtime_s,
            "throughput_tokens_per_s": throughput_tps,
            "throughput_requests_per_s": requests_per_second,
            "avg_latency_s": avg_latency_s,
            "timestamp_utc": datetime.utcnow().isoformat(),
        }
        metrics.update(config.to_metadata())

        return df, metrics
    finally:
        if request_outputs is not None:
            del request_outputs
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_benchmark_artifacts(
    per_prompt_df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Persist benchmark outputs to parquet and JSON files.

    Args:
        per_prompt_df: DataFrame containing prompt-level details.
        metrics: Aggregate metrics dictionary.
        output_dir: Directory to write artifacts into.

    Returns:
        Tuple containing (prompts parquet path, metrics JSON path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{metrics['config_name']}_responses.parquet"
    json_path = output_dir / f"{metrics['config_name']}_metrics.json"

    per_prompt_df.to_parquet(parquet_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return parquet_path, json_path
