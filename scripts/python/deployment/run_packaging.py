#!/usr/bin/env python3
"""Package filtered dataset into SFT format."""

import sys
from pathlib import Path

from scripts.python import project_root

PROJECT_ROOT = project_root(Path(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from nexa_distill.to_sft import run_packaging
import argparse


def main():
    """Package filtered dataset into SFT format."""
    
    filtered_dataset = PROJECT_ROOT / "data/processed/distillation/filtered/filtered_v1.parquet"
    sft_jsonl = PROJECT_ROOT / "data/processed/training/sft_dataset.jsonl"
    sft_parquet = PROJECT_ROOT / "data/processed/training/sft_dataset.parquet"
    
    if not filtered_dataset.exists():
        print(f"❌ ERROR: Filtered dataset not found: {filtered_dataset}")
        print("Please run filtering first: python scripts/python/data_processing/run_filtering.py")
        sys.exit(1)
    
    sft_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("NEXACOMPUTE SFT PACKAGING")
    print("=" * 80)
    print(f"✓ Input: {filtered_dataset.name}")
    print(f"✓ Output JSONL: {sft_jsonl.name}")
    print(f"✓ Output Parquet: {sft_parquet.name}")
    print("=" * 80)
    print()
    
    try:
        args = argparse.Namespace(
            src=filtered_dataset,
            regen=None,
            dst_jsonl=sft_jsonl,
            dst_parquet=sft_parquet,
            prompt_column="user_prompt",
            context_column="context",
            task_type_column="template_name",
            output_column="teacher_output",
        )
        
        print("📦 Packaging dataset into SFT format...")
        run_packaging(args)
        print()
        
        print("=" * 80)
        print("✅ SUCCESS: SFT packaging complete!")
        print("=" * 80)
        print(f"JSONL: {sft_jsonl}")
        print(f"Parquet: {sft_parquet}")
        print("=" * 80)
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR: Packaging failed")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

