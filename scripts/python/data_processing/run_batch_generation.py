#!/usr/bin/env python3
"""Batch teacher generation script with judge evaluation and SampleGate filtering.

This script orchestrates the full data generation pipeline:
1. Generate teacher responses in batches
2. Evaluate with dual-judge system (Judge-F and Judge-R)
3. Apply SampleGate filtering
4. Track metrics and cost
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from tqdm import tqdm

from nexa_distill.utils import OpenAIClient, get_logger
from nexa_eval.rubrics import JUDGE_F_RUBRIC, JUDGE_R_RUBRIC


LOGGER = get_logger(__name__)


@dataclass
class BatchMetrics:
    """Metrics for a generation batch."""
    
    batch_id: int
    timestamp: str
    samples_processed: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    judge_f_mean: float
    judge_r_mean: float
    pass_rate: float
    duration_seconds: float
    samples_per_second: float


class TeacherGenerationPipeline:
    """Orchestrate teacher generation with judging and filtering."""
    
    def __init__(self, config_path: Path, api_key: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.api_key = api_key
        self.client = None
        
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.batch_metrics: List[BatchMetrics] = []
    
    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load generation configuration."""
        with path.open("r") as f:
            return yaml.safe_load(f)
    
    def initialize_client(self) -> None:
        """Initialize OpenAI client."""
        gen_config = self.config["generation"]
        self.client = OpenAIClient(
            model=gen_config["teacher_model"],
            temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"],
            api_key=self.api_key,
        )
    
    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        batch_id: int,
    ) -> tuple[List[Dict[str, Any]], BatchMetrics]:
        """Generate teacher responses for a batch of samples."""
        if not self.client:
            self.initialize_client()
        
        start_time = time.time()
        results = []
        tokens_in = 0
        tokens_out = 0
        
        gen_config = self.config["generation"]
        
        for sample in tqdm(samples, desc=f"Batch {batch_id}", leave=False):
            prompt = self._build_prompt(sample)
            
            try:
                response = self.client.generate(prompt)
                output = response.get("content", "")
                
                sample["teacher_output"] = output
                sample["generation_model"] = gen_config["teacher_model"]
                sample["generation_timestamp"] = datetime.now().isoformat()
                
                tokens_in += response.get("tokens_prompt", 0)
                tokens_out += response.get("tokens_completion", 0)
                
                results.append(sample)
            except Exception as e:
                LOGGER.error(f"Generation failed for sample: {e}")
                continue
        
        duration = time.time() - start_time
        cost = self._calculate_cost(tokens_in, tokens_out)
        
        metrics = BatchMetrics(
            batch_id=batch_id,
            timestamp=datetime.now().isoformat(),
            samples_processed=len(results),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            judge_f_mean=0.0,  # Updated after judging
            judge_r_mean=0.0,
            pass_rate=0.0,
            duration_seconds=duration,
            samples_per_second=len(results) / duration if duration > 0 else 0,
        )
        
        self.total_cost += cost
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        
        return results, metrics
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        metrics: BatchMetrics,
    ) -> List[Dict[str, Any]]:
        """Evaluate samples with dual-judge system."""
        judge_config = self.config["judging"]
        
        judge_f_scores = []
        judge_r_scores = []
        
        for sample in tqdm(samples, desc="Judging", leave=False):
            question = sample.get("prompt_text", "")
            context = sample.get("context", "")
            response = sample.get("teacher_output", "")
            
            try:
                judge_f_result = self._run_judge_f(question, context, response)
                sample["judge_f_response"] = judge_f_result
                judge_f_scores.append(JUDGE_F_RUBRIC.compute_mean_score(judge_f_result))
            except Exception as e:
                LOGGER.error(f"Judge-F failed: {e}")
                sample["judge_f_response"] = None
            
            try:
                judge_r_result = self._run_judge_r(question, context, response)
                sample["judge_r_response"] = judge_r_result
                judge_r_scores.append(JUDGE_R_RUBRIC.compute_mean_score(judge_r_result))
            except Exception as e:
                LOGGER.error(f"Judge-R failed: {e}")
                sample["judge_r_response"] = None
        
        if judge_f_scores:
            metrics.judge_f_mean = sum(judge_f_scores) / len(judge_f_scores)
        if judge_r_scores:
            metrics.judge_r_mean = sum(judge_r_scores) / len(judge_r_scores)
        
        return samples
    
    def _run_judge_f(self, question: str, context: str, response: str) -> Dict[str, Any]:
        """Run Judge-F evaluation."""
        judge_config = self.config["judging"]
        client = OpenAIClient(
            model=judge_config["judge_f_model"],
            temperature=judge_config["judge_temperature"],
            max_tokens=1024,
            api_key=self.api_key,
        )
        
        prompt = JUDGE_F_RUBRIC.format_prompt(question, context, response)
        result = client.generate(
            prompt,
            system_prompt=JUDGE_F_RUBRIC.system_prompt,
        )
        
        content = result.get("content", "{}")
        return json.loads(content)
    
    def _run_judge_r(self, question: str, context: str, response: str) -> Dict[str, Any]:
        """Run Judge-R evaluation."""
        judge_config = self.config["judging"]
        client = OpenAIClient(
            model=judge_config["judge_r_model"],
            temperature=judge_config["judge_temperature"],
            max_tokens=1024,
            api_key=self.api_key,
        )
        
        prompt = JUDGE_R_RUBRIC.format_prompt(question, context, response)
        result = client.generate(
            prompt,
            system_prompt=JUDGE_R_RUBRIC.system_prompt,
        )
        
        content = result.get("content", "{}")
        return json.loads(content)
    
    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        """Build generation prompt from sample."""
        data_config = self.config["data"]
        prompt = sample.get(data_config["prompt_column"], "")
        context = sample.get(data_config["context_column"], "")
        
        if context:
            return f"Context: {context}\n\nQuestion: {prompt}"
        return prompt
    
    def _calculate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost based on token usage."""
        tracking = self.config["tracking"]
        cost_in = (tokens_in / 1000) * tracking["cost_per_1k_input"]
        cost_out = (tokens_out / 1000) * tracking["cost_per_1k_output"]
        return cost_in + cost_out
    
    def save_batch(self, samples: List[Dict[str, Any]], batch_id: int) -> Path:
        """Save batch results to parquet."""
        data_config = self.config["data"]
        output_dir = Path(data_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"generated_batch_{batch_id:04d}.parquet"
        df = pd.DataFrame(samples)
        df.to_parquet(output_path, index=False)
        
        LOGGER.info(f"Saved batch {batch_id} to {output_path}")
        return output_path
    
    def log_metrics(self, metrics: BatchMetrics) -> None:
        """Log batch metrics to manifest."""
        data_config = self.config["data"]
        manifest_path = Path(data_config["manifest_path"])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with manifest_path.open("a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
        
        self.batch_metrics.append(metrics)
    
    def print_summary(self) -> None:
        """Print generation summary."""
        print("\n" + "="*70)
        print("GENERATION SUMMARY")
        print("="*70)
        print(f"Total batches: {len(self.batch_metrics)}")
        print(f"Total cost: ${self.total_cost:.2f}")
        print(f"Total tokens: {self.total_tokens_in + self.total_tokens_out:,}")
        print(f"  Input: {self.total_tokens_in:,}")
        print(f"  Output: {self.total_tokens_out:,}")
        
        if self.batch_metrics:
            avg_judge_f = sum(m.judge_f_mean for m in self.batch_metrics) / len(self.batch_metrics)
            avg_judge_r = sum(m.judge_r_mean for m in self.batch_metrics) / len(self.batch_metrics)
            print(f"\nAverage Judge-F score: {avg_judge_f:.1f}")
            print(f"Average Judge-R score: {avg_judge_r:.1f}")
        
        print("="*70 + "\n")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Batch config YAML")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--skip-judging", action="store_true", help="Skip judge evaluation")
    
    args = parser.parse_args()
    
    pipeline = TeacherGenerationPipeline(args.config, args.api_key)
    
    print(f"Loading dataset from {args.input}")
    df = pd.read_json(args.input, lines=True) if args.input.suffix == ".jsonl" else pd.read_parquet(args.input)
    
    batch_size = args.batch_size or pipeline.config["generation"]["full_batch_size"]
    
    for batch_id in range(args.num_batches):
        start_idx = batch_id * batch_size
        end_idx = start_idx + batch_size
        batch_samples = df.iloc[start_idx:end_idx].to_dict("records")
        
        print(f"\n{'='*70}")
        print(f"Processing Batch {batch_id + 1}/{args.num_batches}")
        print(f"Samples: {len(batch_samples)}")
        print(f"{'='*70}\n")
        
        results, metrics = pipeline.generate_batch(batch_samples, batch_id)
        
        if not args.skip_judging:
            results = pipeline.evaluate_batch(results, metrics)
        
        pipeline.save_batch(results, batch_id)
        pipeline.log_metrics(metrics)
    
    pipeline.print_summary()


if __name__ == "__main__":
    main()

