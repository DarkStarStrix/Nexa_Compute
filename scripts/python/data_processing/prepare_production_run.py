"""Prepare for full production distillation run."""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def archive_old_files(base_dir: Path, archive_dir: Path) -> None:
    """Archive old parquet files to keep workspace clean."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Archive warmup files
    warmup_files = list(base_dir.glob("*warmup*.parquet"))
    warmup_files.extend(list(base_dir.glob("*backup*.parquet")))
    
    if warmup_files:
        print(f"Archiving {len(warmup_files)} old files...")
        for file in warmup_files:
            archive_path = archive_dir / file.name
            shutil.move(str(file), str(archive_path))
            print(f"  Archived: {file.name}")
    else:
        print("No old files to archive")


def verify_inputs(input_path: Path, required_rows: int = None) -> bool:
    """Verify input parquet is ready for production."""
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False
    
    df = pd.read_parquet(input_path)
    print(f"✅ Input file found: {len(df)} rows")
    
    # Check required columns
    required_cols = ['domain', 'template_name', 'user_prompt']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return False
    
    # Check domain/template distribution
    print(f"\nDomain distribution:")
    print(df['domain'].value_counts())
    print(f"\nTemplate distribution:")
    print(df['template_name'].value_counts())
    
    if required_rows and len(df) < required_rows:
        print(f"⚠️  Warning: Only {len(df)} rows, expected {required_rows}")
    
    return True


def prepare_production_run(
    inputs_dir: Path,
    outputs_dir: Path,
    archive_dir: Path,
    warmup_rows: int = 1000,
    full_rows: int = 100000
) -> None:
    """Prepare workspace for production run."""
    print("="*60)
    print("PREPARING PRODUCTION RUN")
    print("="*60)
    
    # Step 1: Archive old output files
    print("\n=== Step 1: Archiving old files ===")
    archive_old_files(outputs_dir, archive_dir)
    
    # Step 2: Verify inputs
    print("\n=== Step 2: Verifying inputs ===")
    input_v1 = inputs_dir / "teacher_inputs_v1.parquet"
    input_v2 = inputs_dir / "teacher_inputs_v2.parquet"
    
    if input_v2.exists():
        print(f"✅ teacher_inputs_v2.parquet exists")
        if verify_inputs(input_v2, full_rows):
            print("✅ Inputs verified for full production run")
    elif input_v1.exists():
        print(f"⚠️  teacher_inputs_v2.parquet not found, using v1 for warmup")
        if verify_inputs(input_v1, warmup_rows):
            print("✅ Inputs verified for warmup run")
            print(f"   Note: For full run, you'll need {full_rows} rows in teacher_inputs_v2.parquet")
    else:
        print("❌ No input files found!")
        return
    
    # Step 3: Verify outputs directory is clean
    print("\n=== Step 3: Verifying outputs directory ===")
    output_files = list(outputs_dir.glob("*.parquet"))
    if output_files:
        print(f"⚠️  Warning: {len(output_files)} parquet files in outputs directory:")
        for f in output_files:
            print(f"   {f.name}")
        print("   (These will be used for sharding during production)")
    else:
        print("✅ Outputs directory is clean")
    
    # Step 4: Verify cleaned directory exists
    print("\n=== Step 4: Verifying cleaned directory ===")
    cleaned_dir = PROJECT_ROOT / "data/processed/distillation/cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Cleaned directory ready: {cleaned_dir}")
    
    # Step 5: Verify SFT directory exists
    print("\n=== Step 5: Verifying SFT directory ===")
    sft_dir = PROJECT_ROOT / "data/processed/distillation/sft_datasets"
    sft_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ SFT directory ready: {sft_dir}")
    
    print("\n" + "="*60)
    print("PREPARATION COMPLETE")
    print("="*60)
    print("\n✅ Ready for warmup run!")
    print("\nNext steps:")
    print("1. Run warmup: python -m nexa_distill.collect_teacher --src <inputs> --max-samples 1000")
    print("2. Verify warmup outputs meet criteria")
    print("3. Run full production: --max-samples 100000")
    print("="*60)


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Prepare for production distillation run")
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/teacher_inputs",
        help="Directory containing teacher input parquets",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/teacher_outputs",
        help="Directory for teacher output parquets",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/archived",
        help="Directory to archive old files",
    )
    parser.add_argument(
        "--warmup-rows",
        type=int,
        default=1000,
        help="Expected rows for warmup",
    )
    parser.add_argument(
        "--full-rows",
        type=int,
        default=100000,
        help="Expected rows for full run",
    )
    
    args = parser.parse_args()
    
    prepare_production_run(
        args.inputs_dir,
        args.outputs_dir,
        args.archive_dir,
        args.warmup_rows,
        args.full_rows
    )


if __name__ == "__main__":
    main()
