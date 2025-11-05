#!/usr/bin/env python3
"""Run filtering pipeline (basic filters + SampleGate)."""

import sys
from pathlib import Path

from scripts.python import project_root

PROJECT_ROOT = project_root(Path(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from nexa_distill.filter_pairs import run_filtering as run_basic_filtering
from nexa_distill.sample_gate import run_sample_gate
import argparse


def main():
    """Run complete filtering pipeline."""
    
    teacher_outputs = PROJECT_ROOT / "data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet"
    basic_filtered = PROJECT_ROOT / "data/processed/distillation/filtered/basic_filtered_v1.parquet"
    sample_gate_filtered = PROJECT_ROOT / "data/processed/distillation/filtered/filtered_v1.parquet"
    rejections = PROJECT_ROOT / "data/processed/distillation/filtered/rejections.parquet"
    
    print("=" * 80)
    print("NEXACOMPUTE FILTERING PIPELINE")
    print("=" * 80)
    print(f"✓ Input: {teacher_outputs.name}")
    print(f"✓ Stage 1: Basic filters → {basic_filtered.name}")
    print(f"✓ Stage 2: SampleGate → {sample_gate_filtered.name}")
    print(f"✓ Rejections: {rejections.name}")
    print("=" * 80)
    print()
    
    basic_filtered.parent.mkdir(parents=True, exist_ok=True)
    sample_gate_filtered.parent.mkdir(parents=True, exist_ok=True)
    
    if not teacher_outputs.exists():
        print(f"❌ ERROR: Teacher outputs not found: {teacher_outputs}")
        print("Please run data generation first: python scripts/python/data_processing/run_full_data_gen.py")
        sys.exit(1)
    
    try:
        print("📊 Stage 1: Running basic heuristic filters...")
        args_basic = argparse.Namespace(
            src=teacher_outputs,
            dst=basic_filtered,
            config=PROJECT_ROOT / "nexa_distill/configs/filters.yaml",
            report=basic_filtered.parent / "basic_filter_report.json",
        )
        run_basic_filtering(args_basic)
        print("✅ Basic filtering complete")
        print()
        
        if not basic_filtered.exists():
            print(f"❌ ERROR: Basic filtered output not found: {basic_filtered}")
            sys.exit(1)
        
        print("🚪 Stage 2: Running SampleGate filters...")
        args_gate = argparse.Namespace(
            src=basic_filtered,
            dst=sample_gate_filtered,
            rejections=rejections,
            min_judge_score=0.80,
            report=sample_gate_filtered.parent / "sample_gate_report.json",
        )
        run_sample_gate(args_gate)
        print("✅ SampleGate filtering complete")
        print()
        
        print("=" * 80)
        print("✅ SUCCESS: Filtering pipeline complete!")
        print("=" * 80)
        print(f"Filtered dataset: {sample_gate_filtered}")
        print(f"Rejections: {rejections}")
        print("=" * 80)
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR: Filtering failed")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

