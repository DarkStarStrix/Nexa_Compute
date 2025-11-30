"""Verify production pipeline is ready for 100k row processing."""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def verify_system_prompts() -> bool:
    """Verify system prompts exist and are correct."""
    print("=== Verifying System Prompts ===")
    
    prompts = {
        'hypothesis': PROJECT_ROOT / 'data/nexa_teacher_hypothesis.txt',
        'methodology': PROJECT_ROOT / 'data/nexa_teacher_methodology.txt',
    }
    
    all_ok = True
    for name, path in prompts.items():
        if path.exists():
            content = path.read_text().lower()
            # Check for key requirements
            has_distilled = 'distilled_response' in content
            has_json = 'json' in content
            has_instruction = 'instruction' in content
            
            if has_distilled and (has_json or has_instruction):
                print(f"✅ {name} prompt exists and has required fields")
            else:
                missing = []
                if not has_distilled: missing.append('distilled_response')
                if not has_json and not has_instruction: missing.append('JSON/instruction')
                print(f"⚠️  {name} prompt exists but missing: {', '.join(missing)}")
                all_ok = False
        else:
            print(f"❌ {name} prompt not found: {path}")
            all_ok = False
    
    return all_ok


def verify_input_data(min_rows: int = 100000) -> bool:
    """Verify input data is ready."""
    print(f"\n=== Verifying Input Data (need {min_rows:,} rows) ===")
    
    input_v2 = PROJECT_ROOT / 'data/processed/distillation/teacher_inputs/teacher_inputs_v2.parquet'
    input_v1 = PROJECT_ROOT / 'data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet'
    
    if input_v2.exists():
        df = pd.read_parquet(input_v2)
        print(f"✅ teacher_inputs_v2.parquet found: {len(df):,} rows")
        
        # Check distribution
        print(f"   Domains: {sorted(df['domain'].unique())}")
        print(f"   Templates: {sorted(df['template_name'].unique())}")
        print(f"   Domain distribution:")
        for domain, count in df['domain'].value_counts().items():
            print(f"     {domain}: {count:,}")
        
        if len(df) >= min_rows:
            print(f"✅ Sufficient rows for production run")
            return True
        else:
            print(f"⚠️  Only {len(df):,} rows, need {min_rows:,}")
            return False
    
    elif input_v1.exists():
        df = pd.read_parquet(input_v1)
        print(f"⚠️  teacher_inputs_v2.parquet not found")
        print(f"   teacher_inputs_v1.parquet exists: {len(df):,} rows (use for warmup)")
        print(f"   Need to create teacher_inputs_v2.parquet with {min_rows:,} rows for full run")
        return False
    else:
        print(f"❌ No input files found")
        return False


def verify_post_processing_script() -> bool:
    """Verify post-processing script exists and is up to date."""
    print("\n=== Verifying Post-Processing Script ===")
    
    script = PROJECT_ROOT / 'scripts/python/data_processing/post_process_distillation.py'
    
    if script.exists():
        content = script.read_text()
        
        # Check for key functions
        required_functions = [
            'clean_hypothesis',
            'clean_methodology',
            'should_drop_sample',
            'extract_distilled_response'
        ]
        
        missing = []
        for func in required_functions:
            if func not in content:
                missing.append(func)
        
        if not missing:
            print(f"✅ Post-processing script exists with all required functions")
            return True
        else:
            print(f"⚠️  Missing functions: {missing}")
            return False
    else:
        print(f"❌ Post-processing script not found: {script}")
        return False


def verify_directories() -> bool:
    """Verify all required directories exist."""
    print("\n=== Verifying Directories ===")
    
    dirs = [
        'data/processed/distillation/teacher_inputs',
        'data/processed/distillation/teacher_outputs',
        'data/processed/distillation/cleaned',
        'data/processed/distillation/sft_datasets',
        'data/processed/distillation/archived',
    ]
    
    all_ok = True
    for dir_path in dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"⚠️  Creating {dir_path}")
            full_path.mkdir(parents=True, exist_ok=True)
            all_ok = False
    
    return all_ok


def estimate_processing_time(rows: int) -> dict:
    """Estimate processing time for large dataset."""
    # Rough estimates based on processing speed
    # Post-processing: ~100-500 rows/second
    # Teacher collection: depends on API rate limits
    
    post_process_speed = 300  # rows/second
    post_process_time = rows / post_process_speed
    
    return {
        'rows': rows,
        'post_process_seconds': post_process_time,
        'post_process_minutes': post_process_time / 60,
        'post_process_hours': post_process_time / 3600,
    }


def main():
    """Main verification."""
    print("="*70)
    print("PRODUCTION PIPELINE VERIFICATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    checks = {
        'System Prompts': verify_system_prompts(),
        'Input Data': verify_input_data(100000),
        'Post-Processing Script': verify_post_processing_script(),
        'Directories': verify_directories(),
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All checks passed! Ready for production run.")
    else:
        print("\n⚠️  Some checks failed. Review issues above before running production.")
    
    # Show processing estimates
    print("\n" + "="*70)
    print("PROCESSING ESTIMATES")
    print("="*70)
    estimates = estimate_processing_time(100000)
    print(f"For {estimates['rows']:,} rows:")
    print(f"  Post-processing: ~{estimates['post_process_minutes']:.1f} minutes")
    print(f"  Teacher collection: Depends on API rate limits and threads")
    
    print("\n" + "="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
