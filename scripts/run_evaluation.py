#!/usr/bin/env python3
"""Run evaluation on distillation data."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexa_eval.evaluate_distillation import run_distillation_evaluation


def main():
    """Run evaluation."""
    
    # Paths
    inputs_path = project_root / "data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet"
    outputs_path = project_root / "data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet"
    report_path = project_root / "data/processed/evaluation/reports/distillation_eval_v1.json"
    
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

