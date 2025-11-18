"""Merge prompts, outputs, and judgments to produce evaluation summaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_PROMPTS = Path("data/processed/evaluation/prompts/prompts.parquet")
DEFAULT_OUTPUT_DIR = Path("data/processed/evaluation/outputs")
DEFAULT_JUDGMENT_DIR = Path("data/processed/evaluation/judgments")
DEFAULT_REPORT_DIR = Path("data/processed/evaluation/reports")
METRIC_COLUMNS = ["correctness", "methodology", "specificity", "clarity", "hallucination_safety"]


@dataclass
class EvaluationData:
    prompts: pd.DataFrame
    outputs: pd.DataFrame
    judgments: pd.DataFrame


def load_outputs(directory: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in directory.glob("outputs_*.parquet"):
        frame = pd.read_parquet(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No output parquet files found in {directory}")
    return pd.concat(frames, ignore_index=True)


def load_judgments(directory: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in directory.glob("judgments_*.parquet"):
        frame = pd.read_parquet(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No judgment parquet files found in {directory}")
    return pd.concat(frames, ignore_index=True)


def merge_data(data: EvaluationData) -> pd.DataFrame:
    merged = data.outputs.merge(data.judgments, on=["id", "model_id"], how="inner")
    merged = merged.merge(data.prompts, on="id", how="left", suffixes=("", "_prompt"))
    merged["overall_score"] = merged[METRIC_COLUMNS].mean(axis=1)
    
    if "raw_response" in merged.columns:
        merged["is_truncated"] = merged["raw_response"].apply(
            lambda x: (
                isinstance(x, dict)
                and x.get("choices", [{}])[0].get("finish_reason") == "length"
            )
            if x is not None
            else False
        )
    else:
        merged["is_truncated"] = False
    
    return merged


def compute_metrics(merged: pd.DataFrame) -> Dict[str, object]:
    per_model = merged.groupby("model_id")[METRIC_COLUMNS + ["overall_score"]].mean().reset_index()
    per_domain = merged.groupby(["model_id", "domain"])[METRIC_COLUMNS].mean().reset_index()
    per_task = merged.groupby(["model_id", "task_type"])[METRIC_COLUMNS].mean().reset_index()
    counts = merged.groupby("model_id")["id"].count().reset_index(name="prompt_count")
    
    truncation_stats = {}
    if "is_truncated" in merged.columns:
        truncation_by_model = merged.groupby("model_id")["is_truncated"].agg(["sum", "count"]).reset_index()
        truncation_by_model["truncation_rate"] = (
            truncation_by_model["sum"] / truncation_by_model["count"] * 100
        ).round(1)
        truncation_stats = truncation_by_model.rename(
            columns={"sum": "truncated_count", "count": "total_count"}
        ).to_dict(orient="records")
    
    return {
        "per_model": per_model.to_dict(orient="records"),
        "per_domain": per_domain.to_dict(orient="records"),
        "per_task_type": per_task.to_dict(orient="records"),
        "counts": counts.to_dict(orient="records"),
        "truncation_stats": truncation_stats,
    }


def plot_per_model(per_model: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    metrics = [col for col in per_model.columns if col != "model_id"]
    for metric in metrics:
        plt.plot(per_model["model_id"], per_model[metric], marker="o", label=metric)
    plt.title("Per-model average scores")
    plt.ylim(1, 5)
    plt.ylabel("Score (1-5)")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse evaluation results.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--judgments-dir", type=Path, default=DEFAULT_JUDGMENT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = EvaluationData(
        prompts=pd.read_parquet(args.prompts),
        outputs=load_outputs(args.outputs_dir),
        judgments=load_judgments(args.judgments_dir),
    )
    merged = merge_data(data)
    metrics = compute_metrics(merged)

    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    per_model_df = pd.DataFrame(metrics["per_model"])
    plot_per_model(per_model_df, plots_dir / "per_model_scores.png")

    merged.to_parquet(args.report_dir / "merged_results.parquet", index=False)
    
    insights = {
        "total_prompts": len(merged),
        "total_models": len(merged["model_id"].unique()),
        "best_overall_model": per_model_df.loc[per_model_df["overall_score"].idxmax(), "model_id"] if not per_model_df.empty else None,
        "best_overall_score": float(per_model_df["overall_score"].max()) if not per_model_df.empty else None,
        "truncation_summary": {
            "total_truncated": int(merged["is_truncated"].sum()) if "is_truncated" in merged.columns else 0,
            "truncation_rate_pct": float(merged["is_truncated"].mean() * 100) if "is_truncated" in merged.columns else 0.0,
        },
        "top_models_by_metric": {}
    }
    
    for metric in METRIC_COLUMNS:
        if metric in per_model_df.columns:
            top_model = per_model_df.loc[per_model_df[metric].idxmax(), "model_id"]
            top_score = float(per_model_df[metric].max())
            insights["top_models_by_metric"][metric] = {
                "model": top_model,
                "score": top_score
            }
    
    insights_path = args.report_dir / "insights.json"
    insights_path.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    
    print(f"[analyze] Summary written to {summary_path}")
    print(f"[analyze] Insights written to {insights_path}")


if __name__ == "__main__":
    main()

