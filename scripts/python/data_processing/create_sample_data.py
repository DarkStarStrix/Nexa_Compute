#!/usr/bin/env python3
"""Create sample scientific corpus for testing the pipeline."""

import argparse
import json
from pathlib import Path
from typing import List


SAMPLE_PROMPTS = [
    {
        "prompt_text": "Explain the mechanism of action of CRISPR-Cas9 gene editing",
        "context": "Molecular biology, genetic engineering",
        "task_type": "explanation",
        "domain": "biology"
    },
    {
        "prompt_text": "Describe the process of nuclear fusion in stars",
        "context": "Astrophysics, stellar evolution",
        "task_type": "explanation",
        "domain": "physics"
    },
    {
        "prompt_text": "What is the relationship between entropy and the second law of thermodynamics?",
        "context": "Thermodynamics, statistical mechanics",
        "task_type": "concept_relationship",
        "domain": "physics"
    },
    {
        "prompt_text": "Explain how neural networks learn through backpropagation",
        "context": "Machine learning, deep learning",
        "task_type": "methodology",
        "domain": "computer_science"
    },
    {
        "prompt_text": "Describe the structure and function of mitochondria",
        "context": "Cell biology, biochemistry",
        "task_type": "explanation",
        "domain": "biology"
    },
    {
        "prompt_text": "What causes superconductivity in certain materials at low temperatures?",
        "context": "Condensed matter physics, quantum mechanics",
        "task_type": "explanation",
        "domain": "physics"
    },
    {
        "prompt_text": "Explain the carbon cycle and its role in climate regulation",
        "context": "Environmental science, biogeochemistry",
        "task_type": "explanation",
        "domain": "environmental_science"
    },
    {
        "prompt_text": "How does the Polymerase Chain Reaction (PCR) amplify DNA?",
        "context": "Molecular biology, biotechnology",
        "task_type": "methodology",
        "domain": "biology"
    },
    {
        "prompt_text": "Describe the principles of quantum entanglement",
        "context": "Quantum mechanics, quantum information",
        "task_type": "explanation",
        "domain": "physics"
    },
    {
        "prompt_text": "What are the key stages of protein folding and why is it important?",
        "context": "Biochemistry, structural biology",
        "task_type": "explanation",
        "domain": "biology"
    }
]


def generate_sample_corpus(num_samples: int, output_path: Path) -> None:
    """Generate a sample scientific corpus by repeating base prompts."""
    
    print(f"Generating {num_samples} sample entries...")
    
    samples: List[dict] = []
    for i in range(num_samples):
        base_prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        sample = base_prompt.copy()
        sample["id"] = f"sample_{i:06d}"
        samples.append(sample)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"✓ Created {len(samples)} samples at {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\nSample entry:")
    print(json.dumps(samples[0], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/sample_corpus.jsonl"),
        help="Output file path"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to generate"
    )
    
    args = parser.parse_args()
    
    generate_sample_corpus(args.samples, args.output)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("\n1. Update batch config to use this file:")
    print(f"   Edit batches/teacher_gen_v1.yaml")
    print(f"   Set: source_dataset: {args.output}")
    print("\n2. Or rename to expected filename:")
    print(f"   mv {args.output} data/raw/scientific_corpus_325M.jsonl")
    print("\n3. Then run QC batch:")
    print("   ./scripts/shell/data_processing/tmux_data_gen.sh 1 100")
    print()


if __name__ == "__main__":
    main()

