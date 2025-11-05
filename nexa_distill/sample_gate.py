"""SampleGate: Quality filtering for teacher-generated samples.

This module implements the SampleGate filtering pipeline that rejects low-quality
samples based on judge scores, JSON validity, and safety flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from nexa_eval.rubrics import JUDGE_F_RUBRIC, JUDGE_R_RUBRIC


@dataclass
class FilterStats:
    """Statistics from filtering operation."""
    
    total_samples: int
    accepted: int
    rejected: int
    
    rejection_reasons: Dict[str, int]
    
    @property
    def acceptance_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.accepted / self.total_samples
    
    @property
    def rejection_rate(self) -> float:
        return 1.0 - self.acceptance_rate


class SampleGate:
    """Filter samples based on quality criteria."""
    
    def __init__(
        self,
        judge_f_threshold: float = 80.0,
        judge_r_threshold: float = 80.0,
        require_both_judges: bool = True,
    ):
        self.judge_f_threshold = judge_f_threshold
        self.judge_r_threshold = judge_r_threshold
        self.require_both_judges = require_both_judges
        
        self.rejection_reasons: Dict[str, int] = {}
    
    def filter_sample(self, sample: Dict[str, Any]) -> tuple[bool, str]:
        """Filter a single sample.
        
        Returns
        -------
        tuple[bool, str]
            (accept, reason) where accept is True if sample passes,
            and reason describes why it was rejected (empty if accepted).
        """
        judge_f = sample.get("judge_f_response")
        judge_r = sample.get("judge_r_response")
        
        if not judge_f and not judge_r:
            return False, "missing_judge_scores"
        
        if judge_f:
            if not isinstance(judge_f, dict):
                try:
                    judge_f = json.loads(judge_f)
                except (json.JSONDecodeError, TypeError):
                    return False, "invalid_judge_f_json"
            
            if not JUDGE_F_RUBRIC.validate_response(judge_f):
                return False, "invalid_judge_f_schema"
            
            judge_f_score = JUDGE_F_RUBRIC.compute_mean_score(judge_f)
            if judge_f_score < self.judge_f_threshold:
                return False, f"judge_f_low_score_{judge_f_score:.1f}"
        
        if judge_r:
            if not isinstance(judge_r, dict):
                try:
                    judge_r = json.loads(judge_r)
                except (json.JSONDecodeError, TypeError):
                    return False, "invalid_judge_r_json"
            
            if not JUDGE_R_RUBRIC.validate_response(judge_r):
                return False, "invalid_judge_r_schema"
            
            judge_r_score = JUDGE_R_RUBRIC.compute_mean_score(judge_r)
            if judge_r_score < self.judge_r_threshold:
                return False, f"judge_r_low_score_{judge_r_score:.1f}"
            
            if judge_r.get("requires_human_review", False):
                return False, "safety_flag_human_review"
            
            safety_flags = judge_r.get("flags", [])
            if any(flag.startswith("safety") for flag in safety_flags):
                if judge_r.get("safety_awareness", 0) < 60:
                    return False, "safety_critical_flag"
        
        if self.require_both_judges and (not judge_f or not judge_r):
            return False, "missing_dual_judge"
        
        return True, ""
    
    def filter_dataset(
        self,
        input_path: Path,
        output_path: Path,
        rejections_path: Optional[Path] = None,
    ) -> FilterStats:
        """Filter a dataset and write accepted/rejected samples.
        
        Parameters
        ----------
        input_path
            Path to input parquet file with judge scores
        output_path
            Path to write accepted samples
        rejections_path
            Optional path to write rejected samples with reasons
        
        Returns
        -------
        FilterStats
            Statistics from the filtering operation
        """
        print(f"Loading dataset from {input_path}")
        df = pd.read_parquet(input_path)
        
        accepted_samples = []
        rejected_samples = []
        rejection_reasons = {}
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering samples"):
            sample = row.to_dict()
            accept, reason = self.filter_sample(sample)
            
            if accept:
                accepted_samples.append(sample)
            else:
                sample["rejection_reason"] = reason
                rejected_samples.append(sample)
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        accepted_df = pd.DataFrame(accepted_samples)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        accepted_df.to_parquet(output_path, index=False)
        print(f"Wrote {len(accepted_samples)} accepted samples to {output_path}")
        
        if rejections_path and rejected_samples:
            rejected_df = pd.DataFrame(rejected_samples)
            rejections_path.parent.mkdir(parents=True, exist_ok=True)
            rejected_df.to_parquet(rejections_path, index=False)
            print(f"Wrote {len(rejected_samples)} rejected samples to {rejections_path}")
        
        stats = FilterStats(
            total_samples=len(df),
            accepted=len(accepted_samples),
            rejected=len(rejected_samples),
            rejection_reasons=rejection_reasons,
        )
        
        return stats
    
    def generate_report(self, stats: FilterStats, output_path: Path) -> None:
        """Generate a filtering report with statistics and visualizations."""
        report_lines = [
            "# SampleGate Filtering Report\n",
            f"## Summary\n",
            f"- Total samples: {stats.total_samples:,}\n",
            f"- Accepted: {stats.accepted:,} ({stats.acceptance_rate:.1%})\n",
            f"- Rejected: {stats.rejected:,} ({stats.rejection_rate:.1%})\n",
            f"\n## Rejection Reasons\n",
        ]
        
        sorted_reasons = sorted(
            stats.rejection_reasons.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for reason, count in sorted_reasons:
            pct = count / stats.total_samples * 100
            report_lines.append(f"- `{reason}`: {count:,} ({pct:.1f}%)\n")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(report_lines))
        print(f"Report written to {output_path}")


def main():
    """CLI entrypoint for SampleGate filtering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter samples with SampleGate")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet file")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file")
    parser.add_argument("--rejections", type=Path, help="Rejections output file")
    parser.add_argument("--report", type=Path, help="Report output file")
    parser.add_argument("--judge-f-threshold", type=float, default=80.0)
    parser.add_argument("--judge-r-threshold", type=float, default=80.0)
    parser.add_argument("--require-both", action="store_true", default=True)
    
    args = parser.parse_args()
    
    gate = SampleGate(
        judge_f_threshold=args.judge_f_threshold,
        judge_r_threshold=args.judge_r_threshold,
        require_both_judges=args.require_both,
    )
    
    stats = gate.filter_dataset(
        input_path=args.input,
        output_path=args.output,
        rejections_path=args.rejections,
    )
    
    print(f"\n{'='*60}")
    print(f"Filtering complete:")
    print(f"  Acceptance rate: {stats.acceptance_rate:.1%}")
    print(f"  Rejected: {stats.rejected:,} samples")
    print(f"{'='*60}\n")
    
    if args.report:
        gate.generate_report(stats, args.report)


if __name__ == "__main__":
    main()

