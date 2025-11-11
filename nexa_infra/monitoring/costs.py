"""Lightweight cost tracking utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def log_cost(
    run_id: str,
    breakdown: Dict[str, float],
    output_dir: Path = Path("runs/manifests"),
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a cost breakdown for a single run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cost_{run_id}.json"
    payload = {
        "run_id": run_id,
        "breakdown": breakdown,
        "total": sum(breakdown.values()),
    }
    if metadata:
        payload["metadata"] = metadata
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


GPU_HOURLY_DEFAULTS: Dict[str, float] = {
    "rtx_4090": 2.25,
    "rtx_5090": 2.95,
    "a100_80gb": 3.75,
    "h100_80gb": 5.5,
}


def estimate_batch_cost(
    *,
    run_id: str,
    nodes: int,
    gpus_per_node: int,
    duration_hours: float,
    gpu_type: Optional[str] = None,
    gpu_hour_cost: Optional[float] = None,
    output_dir: Path = Path("runs/manifests"),
) -> Optional[Path]:
    """Estimate the GPU cost of a batch and log it for later aggregation."""

    if duration_hours <= 0 or nodes <= 0 or gpus_per_node <= 0:
        return None

    rate = gpu_hour_cost
    if rate is None and gpu_type is not None:
        rate = GPU_HOURLY_DEFAULTS.get(gpu_type.replace("-", "_"))
    if rate is None:
        return None

    gpu_hours = nodes * gpus_per_node * duration_hours
    gpu_cost = gpu_hours * rate
    breakdown = {"gpu": round(gpu_cost, 2)}
    metadata = {
        "gpu_hours": round(gpu_hours, 3),
        "gpu_rate_per_hour": rate,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "duration_hours": round(duration_hours, 3),
        "gpu_type": gpu_type,
    }
    return log_cost(run_id, breakdown, output_dir=output_dir, metadata=metadata)


def summarize_costs(manifest_dir: Path) -> None:
    """Aggregate individual cost manifests and print a summary."""

    if not manifest_dir.exists():
        print("[nexa-infra] No cost manifests found")
        return
    totals = []
    for path in manifest_dir.glob("cost_*.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        totals.append(payload)
    if not totals:
        print("[nexa-infra] No cost manifests found")
        return
    grand_total = sum(item["total"] for item in totals)
    print(f"[nexa-infra] Aggregated cost across {len(totals)} runs: ${grand_total:.2f}")
    for item in totals:
        print(f"  - {item['run_id']}: ${item['total']:.2f}")


__all__ = [
    "GPU_HOURLY_DEFAULTS",
    "estimate_batch_cost",
    "log_cost",
    "summarize_costs",
]


