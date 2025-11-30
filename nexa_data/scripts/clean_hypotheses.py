"""Clean hypothesis outputs to keep only first 1-3 sentences, leaving methodology intact."""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_distilled_response(text: str) -> str:
    """Extract the Distilled Response section from teacher output."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Look for **Distilled Response** or **Distilled Response** section
    patterns = [
        r'\*\*Distilled Response\*\*\s*(.+?)(?:\n\n|\*\*|$)',
        r'\*\*Distilled Response\*\*\s*(.+?)$',
        r'Distilled Response\s*(.+?)(?:\n\n|\*\*|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no section found, return the whole text (fallback)
    return text.strip()


def keep_first_sentences(text: str, max_sentences: int = 3) -> str:
    """Keep only the first N sentences from text."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    if not text:
        return ""
    
    # Split into sentences (handle common sentence endings)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # Keep only first max_sentences
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
        # Join and ensure it ends properly
        result = ' '.join(sentences)
        # Remove trailing incomplete sentence markers if any
        result = re.sub(r'\s*\.\.\.\s*$', '', result)
        if not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
    return text


def clean_teacher_outputs(
    input_path: Path,
    output_path: Path,
    max_hypothesis_sentences: int = 3,
    backup: bool = True
) -> None:
    """Clean hypothesis outputs while preserving methodology."""
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Hypothesis rows: {len(df[df['template_name'] == 'hypothesis'])}")
    print(f"Methodology rows: {len(df[df['template_name'] == 'methodology'])}")
    
    # Create backup if requested
    if backup and input_path.exists():
        backup_path = input_path.with_suffix('.parquet.backup')
        print(f"Creating backup: {backup_path}")
        df.to_parquet(backup_path)
    
    # Process only hypothesis templates
    cleaned_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        if row.get('template_name') != 'hypothesis':
            # Skip methodology - keep as is
            continue
        
        teacher_output = row.get('teacher_output', '')
        if pd.isna(teacher_output) or not teacher_output:
            skipped_count += 1
            continue
        
        # Extract distilled response section
        distilled = extract_distilled_response(teacher_output)
        
        if not distilled:
            # If no distilled section found, try to clean the whole output
            distilled = str(teacher_output)
        
        # Keep only first few sentences
        cleaned_distilled = keep_first_sentences(distilled, max_sentences=max_hypothesis_sentences)
        
        if cleaned_distilled:
            # For hypotheses, keep only the streamlined distilled response (no reasoning)
            # This makes hypotheses lean and focused
            new_output = f"**Distilled Response**\n{cleaned_distilled}"
            
            df.at[idx, 'teacher_output'] = new_output
            cleaned_count += 1
        else:
            skipped_count += 1
    
    print(f"\n✅ Cleaned {cleaned_count} hypothesis outputs")
    print(f"⏭️  Skipped {skipped_count} rows")
    print(f"📝 Preserved {len(df[df['template_name'] == 'methodology'])} methodology outputs")
    
    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n💾 Saved cleaned data to: {output_path}")
    print(f"📊 Final row count: {len(df)}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Clean hypothesis outputs to keep only first 1-3 sentences")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet",
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet file path (defaults to overwrite input)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=3,
        help="Maximum sentences to keep in hypothesis outputs",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of input file",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    output_path = args.output if args.output else args.input
    
    clean_teacher_outputs(
        args.input,
        output_path,
        max_hypothesis_sentences=args.max_sentences,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
