"""Evaluate teacher inputs and outputs for distillation quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .judge import Rubric, judge_metrics


def evaluate_teacher_outputs(outputs_path: Path) -> Dict[str, float]:
    """Evaluate teacher outputs for quality metrics."""
    
    df = pd.read_parquet(outputs_path)
    
    metrics = {}
    
    # Basic quality metrics
    if "teacher_output" in df.columns:
        outputs = df["teacher_output"].astype(str)
        
        # Filter out NaN/None values
        valid_outputs = outputs[outputs.notna() & (outputs != "None") & (outputs != "nan")]
        
        if len(valid_outputs) > 0:
            # Length metrics
            lengths = valid_outputs.str.len()
            metrics["avg_output_length"] = float(lengths.mean())
            metrics["min_output_length"] = float(lengths.min())
            metrics["max_output_length"] = float(lengths.max())
            
            # Non-empty outputs
            non_empty = (lengths > 0).sum()
            metrics["completion_rate"] = float(non_empty / len(df))
            metrics["valid_outputs"] = len(valid_outputs)
        else:
            metrics["avg_output_length"] = 0.0
            metrics["completion_rate"] = 0.0
            metrics["valid_outputs"] = 0
        
        # Word count
        word_counts = outputs.str.split().str.len()
        metrics["avg_word_count"] = float(word_counts.mean())
        
        # Check for reasoning sections
        has_reasoning = outputs.str.contains("Reasoning", case=False).sum()
        metrics["has_reasoning_section"] = float(has_reasoning / len(df))
        
        # Check for distilled response sections
        has_distilled = outputs.str.contains("Distilled Response", case=False).sum()
        metrics["has_distilled_section"] = float(has_distilled / len(df))
    
    # Latency metrics
    if "latency_ms" in df.columns:
        metrics["avg_latency_ms"] = float(df["latency_ms"].mean())
        metrics["min_latency_ms"] = float(df["latency_ms"].min())
        metrics["max_latency_ms"] = float(df["latency_ms"].max())
    
    # Token usage metrics
    if "total_tokens" in df.columns:
        metrics["total_tokens_used"] = int(df["total_tokens"].sum())
        metrics["avg_tokens_per_request"] = float(df["total_tokens"].mean())
        metrics["total_cost_estimate"] = float(df["total_tokens"].sum() / 1_000_000 * 2.0)  # $2/1M tokens
    
    if "prompt_tokens" in df.columns and "completion_tokens" in df.columns:
        metrics["total_prompt_tokens"] = int(df["prompt_tokens"].sum())
        metrics["total_completion_tokens"] = int(df["completion_tokens"].sum())
    
    # Domain distribution (if available)
    if "domain" in df.columns:
        domain_counts = df["domain"].value_counts()
        metrics["domain_distribution"] = domain_counts.to_dict()
    
    # Template distribution (if available)
    if "template_name" in df.columns:
        template_counts = df["template_name"].value_counts()
        metrics["template_distribution"] = template_counts.to_dict()
    
    return metrics


def evaluate_teacher_inputs(inputs_path: Path) -> Dict[str, float]:
    """Evaluate teacher inputs for quality metrics."""
    
    df = pd.read_parquet(inputs_path)
    
    metrics = {}
    
    # Basic metrics
    metrics["total_inputs"] = len(df)
    
    # Prompt quality
    if "user_prompt" in df.columns:
        prompts = df["user_prompt"].astype(str)
        lengths = prompts.str.len()
        metrics["avg_prompt_length"] = float(lengths.mean())
        metrics["min_prompt_length"] = float(lengths.min())
        metrics["max_prompt_length"] = float(lengths.max())
        
        # Non-empty prompts
        non_empty = (lengths > 0).sum()
        metrics["valid_prompt_rate"] = float(non_empty / len(df))
    
    # Domain distribution
    if "domain" in df.columns:
        domain_counts = df["domain"].value_counts()
        metrics["domain_distribution"] = domain_counts.to_dict()
        metrics["num_domains"] = len(domain_counts)
    
    # Template distribution
    if "template_name" in df.columns:
        template_counts = df["template_name"].value_counts()
        metrics["template_distribution"] = template_counts.to_dict()
        metrics["num_templates"] = len(template_counts)
    
    return metrics


def run_distillation_evaluation(
    inputs_path: Path,
    outputs_path: Path,
    output_report_path: Path,
) -> Dict[str, float]:
    """Run complete evaluation of distillation pipeline."""
    
    print("=" * 80)
    print("DISTILLATION EVALUATION")
    print("=" * 80)
    print()
    
    # Evaluate inputs
    print("Evaluating teacher inputs...")
    input_metrics = evaluate_teacher_inputs(inputs_path)
    print(f"✓ Loaded {input_metrics.get('total_inputs', 0)} inputs")
    
    # Evaluate outputs
    print("Evaluating teacher outputs...")
    output_metrics = evaluate_teacher_outputs(outputs_path)
    print(f"✓ Loaded {len(pd.read_parquet(outputs_path))} outputs")
    
    # Combine metrics
    all_metrics = {
        "inputs": input_metrics,
        "outputs": output_metrics,
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
    }
    
    # Calculate derived metrics
    if "total_inputs" in input_metrics and "completion_rate" in output_metrics:
        all_metrics["pipeline_completion_rate"] = output_metrics["completion_rate"]
    
    # Save report
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_report_path.open("w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Report saved to: {output_report_path}")
    print()
    print("Key Metrics:")
    print(f"  Inputs: {input_metrics.get('total_inputs', 0)}")
    print(f"  Outputs: {len(pd.read_parquet(outputs_path))}")
    print(f"  Completion Rate: {output_metrics.get('completion_rate', 0):.1%}")
    print(f"  Avg Output Length: {output_metrics.get('avg_output_length', 0):.0f} chars")
    if "avg_latency_ms" in output_metrics:
        print(f"  Avg Latency: {output_metrics['avg_latency_ms']:.0f} ms")
    if "total_cost_estimate" in output_metrics:
        print(f"  Estimated Cost: ${output_metrics['total_cost_estimate']:.2f}")
    print("=" * 80)
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate distillation data")
    parser.add_argument("--inputs", type=Path, required=True, help="Teacher inputs parquet")
    parser.add_argument("--outputs", type=Path, required=True, help="Teacher outputs parquet")
    parser.add_argument("--report", type=Path, required=True, help="Output report JSON path")
    
    args = parser.parse_args()
    
    run_distillation_evaluation(args.inputs, args.outputs, args.report)

