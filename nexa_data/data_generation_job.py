#!/usr/bin/env python3
"""Data generation job script for teacher collection."""

import os
import sys
from pathlib import Path

# Load .env file (including OPENAI_API_PROJECT - it's needed for model access)
project_root = Path(__file__).parent.parent
PROJECT_SLUG = "scientific_assistant"
PROJECT_PROCESSED = project_root / "data/processed" / PROJECT_SLUG
env_path = project_root / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                # Load all env vars including project - it's needed for access
                os.environ[key] = value

# Add project to path
sys.path.insert(0, str(project_root))

    # Import after path setup
from nexa_distill.collect_teacher import run_collection
from nexa_distill.utils.openai_api import OpenAIClient
import argparse


def main():
    """Run data generation job with proper configuration."""
    
    # Load paths from manifest
    manifest_path = PROJECT_PROCESSED / "distillation/manifests/distillation_manifest_v1.json"
    teacher_inputs_path = PROJECT_PROCESSED / "distillation/teacher_inputs/teacher_inputs_v1.parquet"
    system_prompt_path = project_root / "data/system_prompt_template.txt"
    output_path = PROJECT_PROCESSED / "distillation/teacher_outputs/teacher_outputs_v1.parquet"
    
    # Verify files exist
    if not teacher_inputs_path.exists():
        print(f"❌ ERROR: Teacher inputs not found: {teacher_inputs_path}")
        sys.exit(1)
    
    if not system_prompt_path.exists():
        print(f"❌ ERROR: System prompt not found: {system_prompt_path}")
        sys.exit(1)
    
    # Get OpenAI API key (not OpenRouter)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)
    
    # Use gpt-4o-mini (works with project context - confirmed in testing)
    teacher_model = "gpt-4o-mini"
    base_url = None
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("NEXACOMPUTE DATA GENERATION JOB")
    print("=" * 80)
    print(f"✓ Teacher Inputs: {teacher_inputs_path.name}")
    print(f"✓ System Prompt: {system_prompt_path.name}")
    print(f"✓ Manifest: {manifest_path.name if manifest_path.exists() else 'N/A'}")
    print(f"✓ Output: {output_path.name}")
    print(f"✓ Model: {teacher_model}")
    print(f"✓ API: OpenAI")
    print(f"✓ Samples: 100")
    print(f"✓ API Key: {'*' * 20}...{api_key[-4:]}")
    print("=" * 80)
    print()
    
    # Create args namespace
    args = argparse.Namespace(
        src=teacher_inputs_path,
        dst=output_path,
        config=project_root / "nexa_distill/configs/distill_config.yaml",
        teacher=teacher_model,
        system_prompt=system_prompt_path,
        prompt_column="user_prompt",
        context_column="context",
        task_type_column="template_name",
        max_samples=100,
        batch_size=8,
        dry_run=False,
        api_key=api_key,
    )
    
    try:
        print("🚀 Starting teacher collection...")
        print()
        run_collection(args)
        print()
        print("=" * 80)
        print("✅ SUCCESS: Data generation complete!")
        print("=" * 80)
        print(f"Output: {output_path}")
        print(f"Samples generated: 100")
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

