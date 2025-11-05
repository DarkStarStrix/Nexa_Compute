"""Generate 100k teacher inputs from source enhanced JSON files."""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_source_data(domain: str) -> list:
    """Load enhanced JSON data for a domain."""
    source_file = PROJECT_ROOT / f"data/raw/{domain}_enhanced.json"
    if not source_file.exists():
        return []
    
    with open(source_file) as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        return list(data.values()) if data else []
    return data if isinstance(data, list) else []


def create_prompt_variations(context: str, domain: str, template_name: str, num_variations: int = 1) -> list:
    """Create variations of prompts from context."""
    variations = []
    
    # Base prompt templates
    hypothesis_template = "Analyze {aspect} of {topic} in {domain} systems"
    methodology_template = "Design a reproducible experimental methodology to investigate {topic} in {domain}"
    
    # Extract key terms from context
    context_lower = context.lower()
    
    # Simple variation: use different aspects/angles
    aspects = [
        "the evolutionary significance",
        "the mechanistic basis",
        "the functional implications",
        "the regulatory mechanisms",
        "the structural properties",
        "the kinetic behavior",
        "the thermodynamic properties",
        "the molecular interactions",
    ]
    
    for i in range(num_variations):
        if template_name == "hypothesis":
            aspect = random.choice(aspects)
            topic_terms = context.split()[:5]  # First few words as topic
            topic = " ".join(topic_terms)
            prompt = hypothesis_template.format(
                aspect=aspect,
                topic=topic[:50] if len(topic) > 50 else topic,
                domain=domain
            )
        else:  # methodology
            topic_terms = context.split()[:5]
            topic = " ".join(topic_terms)
            prompt = methodology_template.format(
                topic=topic[:50] if len(topic) > 50 else topic,
                domain=domain
            )
        
        variations.append(prompt)
    
    return variations


def generate_teacher_inputs(target_rows: int = 100000) -> pd.DataFrame:
    """Generate teacher inputs dataframe."""
    domains = ['biology', 'materials', 'physics']
    templates = ['hypothesis', 'methodology']
    
    # Load source data
    source_data = {}
    for domain in domains:
        source_data[domain] = load_source_data(domain)
        print(f"Loaded {len(source_data[domain])} items for {domain}")
    
    # Calculate distribution
    rows_per_domain = target_rows // len(domains)
    rows_per_template = rows_per_domain // len(templates)
    
    print(f"\nTarget: {target_rows} rows")
    print(f"Per domain: ~{rows_per_domain} rows")
    print(f"Per domain/template: ~{rows_per_template} rows")
    
    rows = []
    
    for domain in domains:
        domain_sources = source_data[domain]
        if not domain_sources:
            print(f"⚠️  No source data for {domain}, skipping")
            continue
        
        for template_name in templates:
            # Generate variations from source data
            variations_needed = rows_per_template
            
            for _ in range(variations_needed):
                # Select random source item
                source_item = random.choice(domain_sources)
                
                # Extract context (handle different source formats)
                if isinstance(source_item, dict):
                    context = source_item.get('text', source_item.get('content', str(source_item)))
                else:
                    context = str(source_item)
                
                # Create prompt variation
                prompt_variations = create_prompt_variations(context, domain, template_name, num_variations=1)
                user_prompt = prompt_variations[0] if prompt_variations else f"Analyze {context[:100]}"
                
                # Get template and system prompts
                if template_name == "hypothesis":
                    template_prompt = "You are an expert scientific assistant. Given the provided research context, propose 2–3 falsifiable hypotheses that could be evaluated in a wet lab or computational setting. Each hypothesis must:\n- Reference relevant variables or phenomena from the context.\n- Describe a measurable outcome.\n- Avoid speculative claims that cannot be validated with typical lab or simulation resources.\n\nContext:\n{context}"
                    system_prompt = "You are GPT-5 acting as the Nexa scientific teacher. Given an exemplar instruction and dataset context, craft a distillation-ready response that satisfies the requirements for the task template \"hypothesis\". Write in a precise research tone, cite concrete mechanisms when possible, and do not invent unsupported facts."
                else:  # methodology
                    template_prompt = "You are an expert scientific assistant. Design a reproducible experimental methodology to investigate the given research question. The methodology must:\n- Specify the experimental system or model.\n- Define control conditions.\n- Include measurable readouts.\n- Ensure safety and reproducibility.\n\nContext:\n{context}"
                    system_prompt = "You are GPT-5 acting as the Nexa scientific teacher. Given an exemplar instruction and dataset context, craft a distillation-ready response that satisfies the requirements for the task template \"methodology\". Write in a precise research tone, cite concrete mechanisms when possible, and do not invent unsupported facts."
                
                rows.append({
                    'domain': domain,
                    'template_name': template_name,
                    'user_prompt': user_prompt,
                    'template_prompt': template_prompt,
                    'system_prompt': system_prompt,
                    'source_file': f'{domain}_enhanced.json',
                    'generated_at': datetime.now().timestamp(),
                })
    
    df = pd.DataFrame(rows)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Trim to exact target if needed
    if len(df) > target_rows:
        df = df.head(target_rows)
    
    return df


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate 100k teacher inputs")
    parser.add_argument(
        "--target-rows",
        type=int,
        default=100000,
        help="Target number of rows to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/distillation/teacher_inputs/teacher_inputs_v2.parquet",
        help="Output parquet file path",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("GENERATING TEACHER INPUTS")
    print("="*60)
    
    df = generate_teacher_inputs(target_rows=args.target_rows)
    
    print(f"\n✅ Generated {len(df)} rows")
    print(f"\nDistribution by domain:")
    print(df['domain'].value_counts())
    print(f"\nDistribution by template:")
    print(df['template_name'].value_counts())
    print(f"\nDistribution by domain/template:")
    print(df.groupby(['domain', 'template_name']).size())
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\n✅ Saved to: {args.output}")
    print(f"   File size: {args.output.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
