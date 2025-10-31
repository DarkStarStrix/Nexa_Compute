#!/usr/bin/env python3
"""Generate aggregate metrics from manifest files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NexaCompute run manifests")
    parser.add_argument("--manifests", type=Path, default=Path("runs/manifests"), help="Directory containing run manifests")
    parser.add_argument("--output", type=Path, default=Path("runs/manifests/analysis.json"), help="Where to write summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = []
    for path in sorted(args.manifests.glob("*.json")):
        try:
            manifests.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue

    if not manifests:
        args.output.write_text(json.dumps({"runs": 0}, indent=2))
        print("No manifests found")
        return

    train_losses = [m.get("train_result", {}).get("train_loss") for m in manifests if m.get("train_result", {}).get("train_loss") is not None]
    eval_losses = [m.get("eval_metrics", {}).get("eval_loss") for m in manifests if m.get("eval_metrics", {}).get("eval_loss") is not None]
    runtimes = [m.get("train_result", {}).get("train_runtime") for m in manifests if m.get("train_result", {}).get("train_runtime") is not None]

    summary = {
        "runs": len(manifests),
        "avg_train_loss": mean(train_losses) if train_losses else None,
        "avg_eval_loss": mean(eval_losses) if eval_losses else None,
        "avg_runtime_sec": mean(runtimes) if runtimes else None,
        "latest_run": manifests[-1].get("run_id"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Analysis written to {args.output}")


if __name__ == "__main__":
    main()
