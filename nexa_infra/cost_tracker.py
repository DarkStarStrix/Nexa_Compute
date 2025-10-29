"""Lightweight cost tracking utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def log_cost(run_id: str, breakdown: Dict[str, float], output_dir: Path = Path("runs/manifests")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cost_{run_id}.json"
    payload = {"run_id": run_id, "breakdown": breakdown, "total": sum(breakdown.values())}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def summarize_costs(manifest_dir: Path) -> None:
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
