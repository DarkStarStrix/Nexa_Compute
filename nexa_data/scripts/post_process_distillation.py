"""Post-process distillation outputs to create clean, SFT-ready hypotheses/methods."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Constants
MEASURABLE_VARS = [
    "concentration", "expression", "ph", "growth", "lactate",
    "conductivity", "strength", "fluorescence", "atp", "cell count",
    "temperature", "pressure", "voltage", "current", "resistance",
    "absorbance", "optical density", "viability", "proliferation",
    "apoptosis", "metabolism", "enzyme activity", "kinetic",
    "rate", "yield", "efficiency", "density", "mass", "volume"
]

ACTIONABLE_VERBS = [
    "increase", "decrease", "inhibit", "knockdown", "heat", "cool",
    "apply", "measure", "quantify", "assay", "incubate", "simulate",
    "treat", "culture", "expose", "administer", "inject", "add",
    "remove", "extract", "purify", "analyze", "detect", "monitor",
    "record", "evaluate", "assess", "determine"
]

NARRATIVE_FLUFF = [
    "in conclusion", "overall", "this highlights", "based on the above",
    "it is important to note", "furthermore", "additionally", "moreover",
    "therefore", "thus", "hence", "consequently", "in summary",
    "to summarize", "as a result", "in essence", "in other words"
]

UNSAFE_KEYWORDS = [
    "weapon", "pathogen enhancement", "bsl-3", "bsl-4", "biosafety level 3",
    "biosafety level 4", "environmental release", "gene drive", "dual use",
    "gain of function", "crpr on humans", "human germline", "embryo editing"
]


def has_measurable_var(text: str) -> bool:
    """Check if text contains measurable variables."""
    if pd.isna(text) or not text:
        return False
    text_lower = str(text).lower()
    return any(var in text_lower for var in MEASURABLE_VARS)


def has_actionable_verb(text: str) -> bool:
    """Check if text contains actionable verbs."""
    if pd.isna(text) or not text:
        return False
    text_lower = str(text).lower()
    return any(verb in text_lower for verb in ACTIONABLE_VERBS)


def has_narrative_fluff(text: str) -> bool:
    """Check if text contains narrative fluff."""
    if pd.isna(text) or not text:
        return False
    text_lower = str(text).lower()
    return any(fluff in text_lower for fluff in NARRATIVE_FLUFF)


def has_unsafe_content(text: str) -> bool:
    """Check if text contains unsafe content."""
    if pd.isna(text) or not text:
        return False
    text_lower = str(text).lower()
    return any(unsafe in text_lower for unsafe in UNSAFE_KEYWORDS)


def extract_distilled_response(text: str) -> str:
    """Extract the distilled response section from JSON or plain text."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # First, try to extract from JSON structure
    json_match = re.search(r'"distilled_response"\s*:\s*"([^"]+(?:\\.[^"]*)*)"', text, re.DOTALL)
    if json_match:
        # Unescape JSON string
        distilled = json_match.group(1)
        distilled = distilled.replace('\\n', '\n').replace('\\"', '"').replace('\\/', '/')
        return distilled.strip()
    
    # Try unquoted JSON value
    json_match = re.search(r'"distilled_response"\s*:\s*([^,}]+)', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip().strip('"')
    
    # Remove markdown headers if present
    patterns = [
        r'\*\*Distilled Response\*\*\s*',
        r'\*\*Reasoning\*\*\s*',
        r'Distilled Response\s*',
        r'Reasoning\s*',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def clean_hypothesis(text: str, max_sentences: int = 3) -> str:
    """Compress hypothesis to 1-3 sentences per nexa_teacher_hypothesis.txt requirements.
    
    Requirements:
    - Must be falsifiable and measurable
    - Must define a specific variable (gene, catalyst, temperature, protein, pathway, pressure, etc.)
    - Must define a measurable outcome (expression, pH, conductivity, ATP level, etc.)
    - Must propose a testable claim (not just repeat known facts)
    - 1-3 sentences maximum
    """
    if not text or pd.isna(text):
        return ""
    
    # Extract distilled response (remove JSON structure if present)
    text = extract_distilled_response(text)
    
    # Try to extract from JSON if it's structured
    json_match = re.search(r'"distilled_response"\s*:\s*"([^"]+)"', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).replace('\\n', ' ').replace('\\"', '"')
    
    # Remove narrative fluff
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # Filter out fluff and non-testable sentences
    clean_sentences = []
    for sent in sentences:
        # Skip narrative fluff
        if has_narrative_fluff(sent):
            continue
        # Skip sentences that don't have measurable components
        if not (has_measurable_var(sent) or has_actionable_verb(sent)):
            continue
        # Skip sentences that are just facts (no testable claim)
        if any(phrase in sent.lower() for phrase in ["is known", "has been shown", "studies indicate", "it is well established"]):
            continue
        clean_sentences.append(sent)
    
    # Keep only first max_sentences that are falsifiable
    final_sentences = []
    for sent in clean_sentences[:max_sentences * 2]:
        # Check for testable claim indicators
        has_testable = any(word in sent.lower() for word in ["will", "should", "would", "predict", "hypothesis", "if", "then"])
        has_measurable = has_measurable_var(sent)
        
        if has_testable or (has_measurable and has_actionable_verb(sent)):
            final_sentences.append(sent)
            if len(final_sentences) >= max_sentences:
                break
    
    # If we didn't find enough, take first clean sentences
    if len(final_sentences) < max_sentences:
        final_sentences = clean_sentences[:max_sentences]
    
    # Join and ensure proper ending
    result = ' '.join(final_sentences[:max_sentences])
    
    # Ensure proper ending
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    
    return result.strip()


def clean_methodology(text: str, max_lines: int = 6) -> str:
    """Compress methodology to 4-6 lines per nexa_teacher_methodology.txt requirements.
    
    Requirements:
    - Must include experimental system (organism, cell line, chemical sample, material specimen)
    - Must include specific perturbation (inhibitor, knockout, temperature, catalyst, voltage)
    - Must include control group or control condition
    - Must include measurable readout (assay, spectroscopy, fluorescence, conductivity, etc.)
    - Must include safety constraint (BSL-1/2, proper disposal, protective equipment)
    - 4-6 lines maximum
    """
    if not text or pd.isna(text):
        return ""
    
    # Extract distilled response (remove JSON structure if present)
    text = extract_distilled_response(text)
    
    # Try to extract from JSON if it's structured
    json_match = re.search(r'"distilled_response"\s*:\s*"([^"]+)"', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).replace('\\n', '\n').replace('\\"', '"')
    
    # Split by newlines first (methods are often line-separated)
    lines = text.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    
    # If no newlines, split by sentences
    if len(lines) == 1:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        lines = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # Filter and score lines based on methodology requirements
    scored_lines = []
    for line in lines:
        # Skip narrative fluff
        if has_narrative_fluff(line):
            continue
        
        score = 0
        line_lower = line.lower()
        
        # Check for required components
        if any(word in line_lower for word in ['culture', 'cell', 'organism', 'sample', 'specimen', 'material']):
            score += 2  # Experimental system
        if any(word in line_lower for word in ['treat', 'inhibit', 'knockdown', 'add', 'apply', 'expose', 'heat', 'cool']):
            score += 2  # Perturbation
        if any(word in line_lower for word in ['control', 'untreated', 'baseline', 'reference']):
            score += 2  # Control
        if any(word in line_lower for word in ['measure', 'assay', 'detect', 'quantify', 'analyze', 'spectroscopy', 'fluorescence']):
            score += 2  # Measurable readout
        if any(word in line_lower for word in ['bsl', 'safety', 'disposal', 'protective', 'gloves', 'fume hood']):
            score += 2  # Safety constraint
        if has_actionable_verb(line):
            score += 1
        
        if score > 0:
            scored_lines.append((score, line))
    
    # Sort by score and take top lines
    scored_lines.sort(reverse=True, key=lambda x: x[0])
    final_lines = [line for _, line in scored_lines[:max_lines]]
    
    # Ensure we have at least 4 lines if possible
    if len(final_lines) < 4 and len(lines) >= 4:
        # Add high-scoring lines until we have 4
        remaining = [line for line in lines if line not in final_lines]
        final_lines.extend(remaining[:4 - len(final_lines)])
    
    # Join with newlines for step-wise format
    result = '\n'.join(final_lines[:max_lines])
    
    return result.strip()


def is_valid_json_structure(text: str) -> bool:
    """Check if text contains valid JSON structure (even if not parseable)."""
    if not text or pd.isna(text):
        return False
    
    text = str(text).strip()
    
    # Check for JSON-like structure
    has_json_markers = (
        '"instruction"' in text or
        '"distilled_response"' in text or
        '"evidence_trace"' in text or
        (text.startswith('{') and text.endswith('}'))
    )
    
    if not has_json_markers:
        return False
    
    # Try to extract distilled_response even if JSON is malformed
    json_match = re.search(r'"distilled_response"\s*:\s*"([^"]+)"', text, re.DOTALL)
    if json_match:
        return True
    
    # Try parsing if it looks like JSON
    try:
        import json
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json.loads(json_match.group(0))
            return True
    except:
        pass
    
    return False


def should_drop_sample(row: Dict) -> Tuple[bool, str]:
    """Determine if a sample should be dropped and why.
    
    Simple filtering: only drop empty or unsafe content.
    """
    teacher_output = str(row.get('teacher_output', ''))
    
    if not teacher_output or pd.isna(teacher_output) or len(teacher_output.strip()) < 10:
        return True, "empty_output"
    
    # Safety check only
    if has_unsafe_content(teacher_output):
        return True, "unsafe_content"
    
    # Keep everything else - no strict filtering
    return False, ""


def process_row(row: Dict) -> Dict:
    """Process a single row: clean the output based on template type."""
    template_name = row.get('template_name', '')
    teacher_output = row.get('teacher_output', '')
    
    if template_name == 'hypothesis':
        cleaned = clean_hypothesis(teacher_output, max_sentences=3)
    elif template_name == 'methodology':
        cleaned = clean_methodology(teacher_output, max_lines=6)
    else:
        # Unknown template, try hypothesis cleaning
        cleaned = clean_hypothesis(teacher_output, max_sentences=3)
    
    # Enforce length limit (600 chars)
    if len(cleaned) > 600:
        # Truncate at sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        truncated = []
        total_len = 0
        for sent in sentences:
            if total_len + len(sent) + 1 <= 600:
                truncated.append(sent)
                total_len += len(sent) + 1
            else:
                break
        cleaned = ' '.join(truncated)
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
    
    # Create new row with cleaned output
    new_row = row.copy()
    new_row['teacher_output'] = cleaned
    new_row['original_length'] = len(str(teacher_output))
    new_row['cleaned_length'] = len(cleaned)
    
    return new_row


def post_process_distillation(
    input_path: Path,
    output_path: Path,
    dropped_path: Path,
    stats_path: Path
) -> Dict:
    """Main post-processing pipeline."""
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Total input rows: {len(df)}")
    
    stats = {
        'total_input': len(df),
        'dropped': 0,
        'kept': 0,
        'rewritten_count': 0,
        'drop_reasons': {},
        'template_counts': {}
    }
    
    # Step 1: Filter unusable samples
    print("\n=== Step 1: Filtering unusable samples ===")
    kept_rows = []
    dropped_rows = []
    
    for idx, row in df.iterrows():
        should_drop, reason = should_drop_sample(row)
        
        if should_drop:
            dropped_rows.append(row.to_dict())
            stats['dropped'] += 1
            stats['drop_reasons'][reason] = stats['drop_reasons'].get(reason, 0) + 1
        else:
            kept_rows.append(row.to_dict())
            stats['kept'] += 1
    
    print(f"Kept: {stats['kept']}")
    print(f"Dropped: {stats['dropped']}")
    print(f"Drop reasons: {stats['drop_reasons']}")
    
    # Step 2: Compress salvageable outputs
    print("\n=== Step 2: Compressing outputs ===")
    cleaned_rows = []
    
    for row in kept_rows:
        cleaned_row = process_row(row)
        if cleaned_row['cleaned_length'] < cleaned_row['original_length']:
            stats['rewritten_count'] += 1
        cleaned_rows.append(cleaned_row)
    
    print(f"Rewritten: {stats['rewritten_count']}")
    
    # Step 3: Formatting enforcement
    print("\n=== Step 3: Final formatting ===")
    final_rows = []
    
    for row in cleaned_rows:
        cleaned_output = str(row['teacher_output']).strip()
        
        # Just remove any remaining headers, keep everything else
        cleaned_output = extract_distilled_response(cleaned_output)
        row['teacher_output'] = cleaned_output
        row['cleaned_length'] = len(cleaned_output)
        
        final_rows.append(row)
    
    # Step 4: Create output dataframes
    print("\n=== Step 4: Saving outputs ===")
    cleaned_df = pd.DataFrame(final_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    
    # Update stats
    stats['final_count'] = len(cleaned_df)
    for template in cleaned_df['template_name'].unique():
        stats['template_counts'][template] = len(cleaned_df[cleaned_df['template_name'] == template])
    
    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(output_path, index=False)
    print(f"✅ Saved cleaned data: {output_path} ({len(cleaned_df)} rows)")
    
    if len(dropped_df) > 0:
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        dropped_df.to_parquet(dropped_path, index=False)
        print(f"✅ Saved dropped data: {dropped_path} ({len(dropped_df)} rows)")
    
    # Save stats
    import json
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved stats: {stats_path}")
    
    print("\n" + "="*60)
    print("POST-PROCESSING SUMMARY")
    print("="*60)
    print(f"Total input:     {stats['total_input']}")
    print(f"Kept:            {stats['kept']} ({stats['kept']/stats['total_input']*100:.1f}%)")
    print(f"Dropped:         {stats['dropped']} ({stats['dropped']/stats['total_input']*100:.1f}%)")
    print(f"Rewritten:       {stats['rewritten_count']}")
    print(f"Final count:     {stats['final_count']}")
    print(f"\nTemplate distribution:")
    for template, count in stats['template_counts'].items():
        print(f"  {template}: {count}")
    print(f"\nDrop reasons:")
    for reason, count in stats['drop_reasons'].items():
        print(f"  {reason}: {count}")
    print("="*60)
    
    return stats


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Post-process distillation outputs")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet",
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/cleaned/teacher_cleaned.parquet",
        help="Output cleaned parquet file",
    )
    parser.add_argument(
        "--dropped",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/cleaned/teacher_dropped.parquet",
        help="Output dropped samples parquet file",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/cleaned/post_process_stats.json",
        help="Output stats JSON file",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    stats = post_process_distillation(
        args.input,
        args.output,
        args.dropped,
        args.stats
    )
    
    # Exit code based on success
    if stats['final_count'] == 0:
        print("⚠️  Warning: No samples passed post-processing!")
        sys.exit(1)


if __name__ == "__main__":
    main()
