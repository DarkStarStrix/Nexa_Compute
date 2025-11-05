#!/usr/bin/env python3
"""Run evaluation on distillation data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_PATH = Path(__file__).resolve()
REPO_ROOT = CURRENT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.python import project_root

PROJECT_ROOT = project_root(Path(__file__))

from nexa_eval.evaluate_distillation import run_distillation_evaluation


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for distillation evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate distillation teacher data")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet",
        help="Path to teacher inputs parquet",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet",
        help="Path to teacher outputs parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evaluation/reports/distillation_eval_v1.json",
        help="Destination for evaluation report JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation."""

    args = parse_args()

    inputs_path = args.inputs
    outputs_path = args.outputs
    report_path = args.report
    
    # Verify files exist
    if not inputs_path.exists():
        print(f"❌ ERROR: Teacher inputs not found: {inputs_path}")
        sys.exit(1)
    
    if not outputs_path.exists():
        print(f"❌ ERROR: Teacher outputs not found: {outputs_path}")
        print("   Run the data generation job first!")
        sys.exit(1)
    
    # Run evaluation
    run_distillation_evaluation(inputs_path, outputs_path, report_path)


if __name__ == "__main__":
    main()

