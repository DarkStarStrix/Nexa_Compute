"""Validate distillation data integrity before full async runs."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def validate_distillation_data(
    inputs_path: Path,
    outputs_path: Path,
    *,
    min_domains: int = 3,
    min_templates: int = 2,
    min_samples_per_domain: int = 100,
    require_all_domains: bool = False,
    require_all_templates: bool = False,
) -> bool:
    """Validate distillation inputs and outputs for completeness and integrity.
    
    Args:
        inputs_path: Path to teacher inputs parquet
        outputs_path: Path to teacher outputs parquet
        min_domains: Minimum number of domains expected
        min_templates: Minimum number of templates expected
        min_samples_per_domain: Minimum samples per domain/template combo
        require_all_domains: If True, fail if not all input domains are in outputs
        require_all_templates: If True, fail if not all input templates are in outputs
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"Loading inputs from: {inputs_path}")
    inputs_df = pd.read_parquet(inputs_path)
    
    print(f"Loading outputs from: {outputs_path}")
    outputs_df = pd.read_parquet(outputs_path)
    
    print(f"\n=== INPUT VALIDATION ===")
    print(f"Total inputs: {len(inputs_df)}")
    
    # Check required columns
    required_cols = ["domain", "template_name", "user_prompt"]
    missing_cols = [col for col in required_cols if col not in inputs_df.columns]
    if missing_cols:
        print(f"❌ Missing required columns in inputs: {missing_cols}")
        return False
    
    input_domains = inputs_df["domain"].unique()
    input_templates = inputs_df["template_name"].unique()
    
    print(f"Domains: {sorted(input_domains)} ({len(input_domains)} total)")
    print(f"Templates: {sorted(input_templates)} ({len(input_templates)} total)")
    
    # Show input distribution
    input_dist = inputs_df.groupby(["domain", "template_name"]).size().reset_index(name="count")
    print(f"\nInput distribution by domain/template:")
    print(input_dist.to_string(index=False))
    
    print(f"\n=== OUTPUT VALIDATION ===")
    print(f"Total outputs: {len(outputs_df)}")
    
    # Check required columns
    if "teacher_output" not in outputs_df.columns:
        print("❌ Missing 'teacher_output' column in outputs")
        return False
    
    # Check for empty outputs
    empty_outputs = outputs_df["teacher_output"].isna() | (outputs_df["teacher_output"] == "")
    empty_count = empty_outputs.sum()
    if empty_count > 0:
        print(f"⚠️  Warning: {empty_count} empty outputs found")
    
    output_domains = outputs_df["domain"].unique() if "domain" in outputs_df.columns else []
    output_templates = outputs_df["template_name"].unique() if "template_name" in outputs_df.columns else []
    
    print(f"Domains: {sorted(output_domains)} ({len(output_domains)} total)")
    print(f"Templates: {sorted(output_templates)} ({len(output_templates)} total)")
    
    # Show output distribution
    if "domain" in outputs_df.columns and "template_name" in outputs_df.columns:
        output_dist = outputs_df.groupby(["domain", "template_name"]).size().reset_index(name="count")
        print(f"\nOutput distribution by domain/template:")
        print(output_dist.to_string(index=False))
    
    # Validate domain coverage
    print(f"\n=== DOMAIN COVERAGE ===")
    if require_all_domains:
        missing_domains = set(input_domains) - set(output_domains)
        if missing_domains:
            print(f"❌ Missing domains in outputs: {sorted(missing_domains)}")
            return False
    
    if len(output_domains) < min_domains:
        print(f"❌ Insufficient domain coverage: {len(output_domains)} < {min_domains}")
        return False
    
    # Validate template coverage
    print(f"\n=== TEMPLATE COVERAGE ===")
    if require_all_templates:
        missing_templates = set(input_templates) - set(output_templates)
        if missing_templates:
            print(f"❌ Missing templates in outputs: {sorted(missing_templates)}")
            return False
    
    if len(output_templates) < min_templates:
        print(f"❌ Insufficient template coverage: {len(output_templates)} < {min_templates}")
        return False
    
    # Validate sample counts per domain/template
    print(f"\n=== SAMPLE COUNT VALIDATION ===")
    if "domain" in outputs_df.columns and "template_name" in outputs_df.columns:
        valid_outputs = outputs_df[outputs_df["teacher_output"].notna() & (outputs_df["teacher_output"] != "")]
        sample_counts = valid_outputs.groupby(["domain", "template_name"]).size()
        
        insufficient = []
        for (domain, template), count in sample_counts.items():
            if count < min_samples_per_domain:
                insufficient.append((domain, template, count, min_samples_per_domain))
        
        if insufficient:
            print(f"⚠️  Warning: Some domain/template combos have fewer than {min_samples_per_domain} samples:")
            for domain, template, count, min_required in insufficient:
                print(f"  {domain}/{template}: {count} < {min_required}")
            # Don't fail, just warn
    
    # Validate merge compatibility
    print(f"\n=== MERGE COMPATIBILITY ===")
    inputs_df = inputs_df.copy()
    outputs_df = outputs_df.copy()
    
    inputs_df["_merge_key"] = (
        inputs_df["domain"].astype(str) + "|||" + 
        inputs_df["template_name"].astype(str) + "|||" + 
        inputs_df["user_prompt"].astype(str)
    )
    
    if "domain" in outputs_df.columns and "template_name" in outputs_df.columns:
        outputs_df["_merge_key"] = (
            outputs_df["domain"].astype(str) + "|||" + 
            outputs_df["template_name"].astype(str) + "|||" + 
            outputs_df["user_prompt"].astype(str)
        )
        
        # Check for duplicate merge keys
        input_dupes = inputs_df["_merge_key"].duplicated().sum()
        output_dupes = outputs_df["_merge_key"].duplicated().sum()
        
        if input_dupes > 0:
            print(f"⚠️  Warning: {input_dupes} duplicate prompts in inputs")
        if output_dupes > 0:
            print(f"⚠️  Warning: {output_dupes} duplicate prompts in outputs")
        
        # Test merge
        combined = inputs_df.merge(
            outputs_df[["_merge_key", "teacher_output"]],
            on="_merge_key",
            how="left"
        )
        
        matched_count = combined["teacher_output"].notna().sum()
        match_rate = (matched_count / len(combined) * 100) if len(combined) > 0 else 0
        
        print(f"Merge test: {matched_count}/{len(combined)} rows matched ({match_rate:.1f}%)")
        
        if matched_count == 0:
            print("❌ No rows matched - merge keys incompatible")
            return False
    
    print(f"\n✅ Validation passed!")
    return True


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Validate distillation data integrity")
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
        "--min-domains",
        type=int,
        default=3,
        help="Minimum number of domains required",
    )
    parser.add_argument(
        "--min-templates",
        type=int,
        default=2,
        help="Minimum number of templates required",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples per domain/template combo",
    )
    parser.add_argument(
        "--require-all-domains",
        action="store_true",
        help="Fail if not all input domains are present in outputs",
    )
    parser.add_argument(
        "--require-all-templates",
        action="store_true",
        help="Fail if not all input templates are present in outputs",
    )
    
    args = parser.parse_args()
    
    if not args.inputs.exists():
        print(f"❌ Inputs file not found: {args.inputs}")
        sys.exit(1)
    
    if not args.outputs.exists():
        print(f"❌ Outputs file not found: {args.outputs}")
        sys.exit(1)
    
    success = validate_distillation_data(
        args.inputs,
        args.outputs,
        min_domains=args.min_domains,
        min_templates=args.min_templates,
        min_samples_per_domain=args.min_samples,
        require_all_domains=args.require_all_domains,
        require_all_templates=args.require_all_templates,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
