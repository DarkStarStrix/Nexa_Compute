"""Serve a Streamlit leaderboard UI for viewing evaluation data and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


def load_evaluation_data() -> pd.DataFrame:
    """Load evaluation reports from organized data structure."""
    eval_dir = Path("data/processed/evaluation/reports")
    if not eval_dir.exists():
        return pd.DataFrame()
    
    reports = []
    for report_path in eval_dir.glob("eval_report_*.json"):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "metrics" in data:
                row = {"run_id": report_path.stem.replace("eval_report_", ""), **data["metrics"]}
                reports.append(row)
        except (json.JSONDecodeError, KeyError):
            continue
    
    return pd.DataFrame(reports)


def load_distillation_data() -> pd.DataFrame:
    """Load distillation teacher inputs for visualization."""
    from nexa_data.data_analysis.query_data import DataQuery
    
    try:
        query = DataQuery()
        return query.get_teacher_inputs(version="v1")
    except FileNotFoundError:
        return pd.DataFrame()


def load_training_data() -> pd.DataFrame:
    """Load training dataset statistics."""
    train_dir = Path("data/processed/training")
    if not train_dir.exists():
        return pd.DataFrame()
    
    # Look for training metadata or statistics
    stats_files = list(train_dir.rglob("*.json"))
    if not stats_files:
        return pd.DataFrame()
    
    # Load first available stats file
    try:
        with open(stats_files[0]) as f:
            data = json.load(f)
        return pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame()
    except (json.JSONDecodeError, KeyError):
        return pd.DataFrame()


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="NexaCompute Dashboard", layout="wide")
    st.title("NexaCompute Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Evaluation", "Distillation", "Training"])
    
    with tab1:
        st.header("Evaluation Leaderboard")
        eval_df = load_evaluation_data()
        if eval_df.empty:
            st.info("No evaluation reports found. Run evaluation first.")
        else:
            st.dataframe(eval_df, use_container_width=True)
            
            # Metric charts
            if len(eval_df) > 0:
                metric_cols = [col for col in eval_df.columns if col != "run_id"]
                if metric_cols:
                    selected_metric = st.selectbox("Select metric", metric_cols)
                    st.bar_chart(eval_df.set_index("run_id")[selected_metric])
    
    with tab2:
        st.header("Distillation Data")
        distill_df = load_distillation_data()
        if distill_df.empty:
            st.info("No distillation data found. Run data analysis notebook first.")
        else:
            st.dataframe(distill_df.head(100), use_container_width=True)
            st.metric("Total Teacher Inputs", len(distill_df))
            
            # Domain distribution
            if "domain" in distill_df.columns:
                domain_counts = distill_df["domain"].value_counts()
                st.bar_chart(domain_counts)
    
    with tab3:
        st.header("Training Data")
        train_df = load_training_data()
        if train_df.empty:
            st.info("No training data statistics found.")
        else:
            st.dataframe(train_df, use_container_width=True)


def serve_leaderboard(host: Optional[str] = None, port: int = 8080) -> None:
    """Serve the Streamlit leaderboard UI."""
    import subprocess
    import sys
    
    script_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(script_path), "--server.port", str(port)]
    if host:
        cmd.extend(["--server.address", host])
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
