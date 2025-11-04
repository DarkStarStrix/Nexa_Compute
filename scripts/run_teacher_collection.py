#!/usr/bin/env python3
"""Run teacher collection with progress tracking."""

import os
import sys
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    # Manual load (more reliable)
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ[key] = value
else:
    # Try dotenv as fallback
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import and run
from nexa_distill.collect_teacher import run_collection, parse_args
import argparse

def main():
    """Run teacher collection with 100 samples."""
    
    # Check API key (try multiple possible names)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("❌ ERROR: OpenAI API key not found")
        print("Checked environment variables: OPENAI_API_KEY, OPENAI_KEY, API_KEY")
        print("\nAvailable env vars with 'API' or 'KEY':")
        for key, value in os.environ.items():
            if "API" in key.upper() or "KEY" in key.upper():
                print(f"  {key}: {'*' * 20}...{str(value)[-4:] if len(str(value)) > 4 else '***'}")
        print("\nPlease ensure .env contains: OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    print("=" * 70)
    print("NEXACOMPUTE TEACHER COLLECTION")
    print("=" * 70)
    print(f"✓ API Key: {'*' * 20}...{api_key[-4:]}")
    print(f"✓ Model: gpt-4o-mini")
    print(f"✓ Samples: 100")
    print("=" * 70)
    print()
    
    # Build arguments
    src = project_root / "data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet"
    dst = project_root / "data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet"
    system_prompt = project_root / "data/system_prompt_template.txt"
    
    # Ensure output directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Create args namespace
    args = argparse.Namespace(
        src=src,
        dst=dst,
        config=project_root / "nexa_distill/configs/distill_config.yaml",
        teacher="gpt-4o-mini",
        system_prompt=system_prompt,
        prompt_column="user_prompt",
        context_column="context",
        task_type_column="template_name",
        max_samples=100,
        batch_size=8,
        dry_run=False,
        api_key=api_key,
    )
    
    try:
        run_collection(args)
        print()
        print("=" * 70)
        print("✅ SUCCESS: Teacher collection complete!")
        print("=" * 70)
        print(f"Output saved to: {dst}")
        print(f"Total samples: 100")
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR: Teacher collection failed")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

