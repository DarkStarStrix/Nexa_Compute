#!/usr/bin/env python3
"""Full-scale data generation job script for teacher collection."""

import os
import sys
from pathlib import Path

from scripts.python import project_root

PROJECT_ROOT = project_root(Path(__file__))
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ[key] = value

sys.path.insert(0, str(PROJECT_ROOT))

from nexa_distill.collect_teacher import run_collection
import argparse


def main():
    """Run full-scale data generation job."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)
    
    teacher_inputs_path = PROJECT_ROOT / "data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet"
    system_prompt_path = PROJECT_ROOT / "data/system_prompt_template.txt"
    output_path = PROJECT_ROOT / "data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet"
    
    if not teacher_inputs_path.exists():
        print(f"❌ ERROR: Teacher inputs not found: {teacher_inputs_path}")
        sys.exit(1)
    
    if not system_prompt_path.exists():
        print(f"❌ ERROR: System prompt not found: {system_prompt_path}")
        sys.exit(1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("NEXACOMPUTE FULL-SCALE DATA GENERATION")
    print("=" * 80)
    print(f"✓ Teacher Inputs: {teacher_inputs_path.name}")
    print(f"✓ System Prompt: {system_prompt_path.name}")
    print(f"✓ Output: {output_path.name}")
    print(f"✓ Model: gpt-4o-mini")
    print(f"✓ Batch Size: 8")
    print(f"✓ Max Samples: None (full dataset)")
    print("=" * 80)
    print()
    
    args = argparse.Namespace(
        src=teacher_inputs_path,
        dst=output_path,
        config=PROJECT_ROOT / "nexa_distill/configs/distill_config.yaml",
        teacher="gpt-4o-mini",
        system_prompt=system_prompt_path,
        prompt_column="user_prompt",
        context_column="context",
        task_type_column="template_name",
        max_samples=None,
        batch_size=8,
        dry_run=False,
        api_key=api_key,
    )
    
    try:
        print("🚀 Starting full-scale teacher collection...")
        print()
        run_collection(args)
        print()
        print("=" * 80)
        print("✅ SUCCESS: Data generation complete!")
        print("=" * 80)
        print(f"Output: {output_path}")
        print("=" * 80)
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR: Data generation failed")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

