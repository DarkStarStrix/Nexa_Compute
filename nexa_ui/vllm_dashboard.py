"""Streamlit app to explore vLLM benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "vllm_benchmarks"


@st.cache_data(show_spinner=False)
def load_metrics_from_dirs(results_root: Path) -> pd.DataFrame:
    """Aggregate all *_metrics.json files into a single DataFrame."""
    records: List[Dict[str, object]] = []
    for metrics_path in results_root.glob("*/*_metrics.json"):
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        data.setdefault("config_name", metrics_path.parent.name)
        records.append(data)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("config_name").reset_index(drop=True)
    numeric_cols = [
        "throughput_tokens_per_s",
        "throughput_requests_per_s",
        "avg_latency_s",
        "runtime_s",
        "total_output_tokens",
        "num_requests",
        "gpu_memory_utilization",
        "max_model_len",
        "max_new_tokens",
        "max_num_seqs",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_responses_for_config(config_name: str) -> Optional[pd.DataFrame]:
    """Load the response parquet for a single configuration."""
    config_dir = RESULTS_ROOT / config_name
    parquet_files = sorted(config_dir.glob("*_responses.parquet"))
    if parquet_files:
        return pd.read_parquet(parquet_files[0])
    results_parquet = RESULTS_ROOT / "results.parquet"
    if results_parquet.exists():
        results_df = pd.read_parquet(results_parquet)
        return results_df[results_df["config_name"] == config_name]
    return None


def main() -> None:
    st.set_page_config(page_title="Nexa Compute vLLM Benchmarks", layout="wide")
    st.title("vLLM Benchmark Dashboard")
    st.caption(
        "Unified view of all inference experiments: review the exact questions and "
        "responses, then inspect configuration and performance analytics."
    )

    metrics_df = load_metrics_from_dirs(RESULTS_ROOT)
    if metrics_df.empty:
        st.warning(
            "No benchmark metrics found under "
            f"`{RESULTS_ROOT}`. Run the benchmark script first."
        )
        return

    metrics_df = metrics_df.fillna({"config_name": "unknown"})
    config_names = metrics_df["config_name"].tolist()

    tab_qna, tab_configs = st.tabs(["Q&A Explorer", "Configuration & Analysis"])

    with tab_qna:
        st.subheader("Q&A Explorer")
        selected_config = st.selectbox(
            "Select configuration", config_names, key="qna_select"
        )
        responses_df = load_responses_for_config(selected_config)
        if responses_df is None or responses_df.empty:
            st.info(
                "No per-prompt responses available for this configuration. "
                "Ensure the benchmark saved prompt-level outputs."
            )
        else:
            responses_df = responses_df.reset_index(drop=True)
            prompt_options = [f"Prompt {idx + 1}" for idx in responses_df.index]
            view_mode = st.radio(
                "Display mode",
                ["Single prompt", "Show all"],
                horizontal=True,
                key="qna_view_mode",
            )

            def render_prompt(idx: int) -> None:
                prompt_row = responses_df.loc[idx]
                st.markdown(f"#### Prompt {idx + 1}")
                st.markdown(f"**Question**\n\n{prompt_row['prompt']}")
                st.markdown(f"**Answer**\n\n{prompt_row['response']}")
                tokens = int(prompt_row.get("completion_tokens", 0))
                st.caption(f"Completion tokens: {tokens}")

            if view_mode == "Single prompt":
                selected_prompt = st.selectbox(
                    "Select prompt", prompt_options, key="prompt_select"
                )
                idx = prompt_options.index(selected_prompt)
                render_prompt(idx)
            else:
                for idx in responses_df.index:
                    with st.expander(prompt_options[idx], expanded=False):
                        render_prompt(idx)

            st.info(
                "If an answer appears truncated, rerun the benchmark with a larger "
                "`max_new_tokens` value—current runs cap completions at 192–256 tokens."
            )

    with tab_configs:
        st.subheader("Configuration Overview")
        selected_summary_config = st.selectbox(
            "Select configuration", config_names, key="config_select"
        )
        config_cols = [
            "config_name",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "max_model_len",
            "max_new_tokens",
            "max_num_seqs",
            "dtype",
            "timestamp_utc",
        ]
        available_cols = [col for col in config_cols if col in metrics_df.columns]
        st.dataframe(metrics_df[available_cols], use_container_width=True)

        st.markdown("**Selected Configuration Details**")
        selected_row = metrics_df[
            metrics_df["config_name"] == selected_summary_config
        ].iloc[0]
        detail_cols = [
            "model_id",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "max_model_len",
            "max_new_tokens",
            "max_num_seqs",
            "dtype",
            "temperature",
            "top_p",
            "timestamp_utc",
        ]
        detail_payload = {
            field: selected_row.get(field)
            for field in detail_cols
            if field in selected_row and pd.notnull(selected_row.get(field))
        }
        st.json(detail_payload)

        st.subheader("Performance Comparison")
        perf_cols = [
            "config_name",
            "throughput_tokens_per_s",
            "throughput_requests_per_s",
            "avg_latency_s",
        ]
        perf_cols = [col for col in perf_cols if col in metrics_df.columns]
        perf_df = metrics_df[perf_cols].set_index("config_name")
        perf_plot_df = perf_df.rename(
            columns={
                "throughput_tokens_per_s": "Tokens / s",
                "throughput_requests_per_s": "Requests / s",
                "avg_latency_s": "Avg Latency (s)",
            }
        )
        st.bar_chart(perf_plot_df[["Tokens / s", "Requests / s"]], use_container_width=True)

        st.subheader("Notes")
        notes = []
        if not perf_plot_df.empty:
            best_throughput = perf_plot_df["Tokens / s"].idxmax()
            best_value = perf_plot_df.loc[best_throughput, "Tokens / s"]
            notes.append(
                f"- Highest throughput: `{best_throughput}` ({best_value:.1f} tokens/s)."
            )
            worst_latency = perf_plot_df["Avg Latency (s)"].idxmax()
            worst_value = perf_plot_df.loc[worst_latency, "Avg Latency (s)"]
            notes.append(
                f"- Longest latency: `{worst_latency}` ({worst_value:.3f} s avg)."
            )
            notes.append(
                "- TP=1 configs are memory-stable; TP=2 or INT4 experiments require "
                "resetting GPU0 or using quantized checkpoints."
            )
        else:
            notes.append("- Performance metrics unavailable.")
        for note in notes:
            st.markdown(note)


if __name__ == "__main__":
    main()

