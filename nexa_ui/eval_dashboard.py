"""Streamlit dashboard for scientific evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import altair as alt
import pandas as pd
import streamlit as st

RESULTS_DIR = Path("results/evaluation")
MERGED_RESULTS_PATH = RESULTS_DIR / "merged_results.parquet"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
INSIGHTS_PATH = RESULTS_DIR / "insights.json"


@st.cache_data(show_spinner=False)
def _load_merged_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    merged = pd.read_parquet(path)
    if "is_truncated" not in merged.columns and "raw_response" in merged.columns:
        merged["is_truncated"] = merged["raw_response"].apply(
            lambda x: (
                isinstance(x, dict)
                and x.get("choices", [{}])[0].get("finish_reason") == "length"
            )
            if x is not None
            else False
        )
    return merged


@st.cache_data(show_spinner=False)
def _load_summary(path: Path) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def _load_insights(path: Path) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _check_truncation(row: pd.Series) -> bool:
    if "is_truncated" in row and row["is_truncated"]:
        return True
    if "raw_response" in row and row["raw_response"] is not None:
        raw = row["raw_response"]
        if isinstance(raw, dict):
            choice = raw.get("choices", [{}])[0]
            return choice.get("finish_reason") == "length"
    return False


def render_qna_tab(merged: pd.DataFrame) -> None:
    st.subheader("Q&A Explorer")
    if merged.empty:
        st.info("No merged evaluation data found.")
        return

    models = sorted(merged["model_id"].unique())
    selected_model = st.selectbox("Select Model", models, key="qna_model_select")

    filtered = merged[merged["model_id"] == selected_model].reset_index(drop=True)

    domain_options = sorted(filtered["domain"].dropna().unique())
    if domain_options:
        selected_domain = st.selectbox(
            "Filter by Domain", ["All"] + domain_options, key="qna_domain_select"
        )
        if selected_domain != "All":
            filtered = filtered[filtered["domain"] == selected_domain]

    view_mode = st.radio(
        "Display Mode",
        ["Single Prompt", "Show All"],
        horizontal=True,
        key="qna_view_mode",
    )

    def render_prompt_row(idx: int, row: pd.Series) -> None:
        is_truncated = _check_truncation(row)
        truncation_badge = "⚠️ **TRUNCATED**" if is_truncated else ""

        st.markdown(f"#### Prompt {idx + 1} {truncation_badge}")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Question**")
            st.markdown(row.get("prompt", ""))
        with col2:
            if "domain" in row and pd.notna(row["domain"]):
                st.metric("Domain", row["domain"])
            if "task_type" in row and pd.notna(row["task_type"]):
                st.caption(f"Task: {row['task_type']}")

        if "system_prompt" in row and pd.notna(row.get("system_prompt")):
            with st.expander("System Prompt", expanded=False):
                st.text(row["system_prompt"])

        st.markdown("**Model Answer**")
        output_text = row.get("output", "")
        if is_truncated:
            st.warning(
                "⚠️ This response was truncated due to max_tokens limit. "
                "The full response may be incomplete."
            )
        st.markdown(output_text)

        st.markdown("**Judge Scores**")
        metrics = [
            "correctness",
            "methodology",
            "specificity",
            "clarity",
            "hallucination_safety",
        ]
        cols = st.columns(5)
        for col, metric in zip(cols, metrics):
            score = row.get(metric, 0)
            col.metric(
                metric.replace("_", " ").title(),
                f"{score:.1f}",
                delta=None if score == 0 else None,
            )

        if "comments" in row and pd.notna(row.get("comments")):
            with st.expander("Judge Comments", expanded=False):
                st.text(row["comments"])

        if "tokens_out" in row:
            st.caption(f"Output tokens: {int(row['tokens_out'])}")

    if view_mode == "Single Prompt":
        prompt_options = [
            f"Prompt {idx + 1}" for idx in range(len(filtered))
        ]
        selected_prompt = st.selectbox(
            "Select Prompt", prompt_options, key="prompt_select"
        )
        idx = prompt_options.index(selected_prompt)
        render_prompt_row(idx, filtered.iloc[idx])
    else:
        for idx in range(len(filtered)):
            with st.expander(
                f"Prompt {idx + 1}",
                expanded=False,
            ):
                render_prompt_row(idx, filtered.iloc[idx])


def render_scores_tab(summary: Dict[str, object]) -> None:
    st.subheader("Model Performance Scores")
    per_model = summary.get("per_model")
    if not per_model:
        st.info("Run the analysis script to populate summary.json.")
        return
    df = pd.DataFrame(per_model)
    if df.empty:
        return

    df = df.copy()
    
    truncation_stats = summary.get("truncation_stats", [])
    if truncation_stats:
        trunc_df = pd.DataFrame(truncation_stats)
        df = df.merge(
            trunc_df[["model_id", "truncated_count", "total_count", "truncation_rate"]],
            on="model_id",
            how="left"
        )
        df["truncation_status"] = df["truncation_rate"].apply(
            lambda x: "⚠️ Truncated" if pd.notna(x) and x > 0 else "✓ Complete"
        )
    
    if "overall_score" in df.columns:
        df = df.sort_values(by="overall_score", ascending=False).reset_index(drop=True)

    st.markdown("#### Summary Table")
    display_df = df.copy()
    
    column_order = ["model_id"]
    if "truncation_status" in display_df.columns:
        column_order.append("truncation_status")
    if "truncation_rate" in display_df.columns:
        column_order.append("truncation_rate")
    if "overall_score" in display_df.columns:
        column_order.append("overall_score")
    metric_cols = ["correctness", "methodology", "specificity", "clarity", "hallucination_safety"]
    for col in metric_cols:
        if col in display_df.columns:
            column_order.append(col)
    if "truncated_count" in display_df.columns:
        column_order.extend(["truncated_count", "total_count"])
    
    display_df = display_df[[col for col in column_order if col in display_df.columns]]
    
    for col in display_df.columns:
        if col not in ["model_id", "truncation_status"] and display_df[col].dtype in ["float64", "float32"]:
            display_df[col] = display_df[col].round(2)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    color_domain = df["model_id"].tolist()
    color_scale = alt.Scale(domain=color_domain, scheme="category20")

    metric_columns = [
        column
        for column in [
            "correctness",
            "methodology",
            "specificity",
            "clarity",
            "hallucination_safety",
        ]
        if column in df.columns
    ]
    has_overall = "overall_score" in df.columns
    has_metrics = bool(metric_columns)

    if not has_overall and not has_metrics:
        return

    st.markdown("---")
    st.markdown("#### Overall Performance")
    
    if has_overall:
        chart_overall = (
            alt.Chart(df)
            .mark_bar(size=50, cornerRadius=4)
            .encode(
                x=alt.X(
                    "model_id:N",
                    sort=alt.SortField(field="overall_score", order="descending"),
                    title="Model",
                    axis=alt.Axis(labelAngle=-45, labelLimit=200),
                ),
                y=alt.Y(
                    "overall_score:Q",
                    title="Overall Score",
                    scale=alt.Scale(domain=[0, 5]),
                ),
                color=alt.Color(
                    "model_id:N",
                    scale=color_scale,
                    legend=alt.Legend(title="Model", columns=2),
                ),
                tooltip=[
                    alt.Tooltip("model_id:N", title="Model"),
                    alt.Tooltip("overall_score:Q", title="Overall Score", format=".2f"),
                ],
            )
            .properties(height=400, width=600)
        )
        st.altair_chart(chart_overall, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Performance by Metric")
    
    if has_metrics:
        metric_title_map = {
            "correctness": "Correctness",
            "methodology": "Methodology",
            "specificity": "Specificity",
            "clarity": "Clarity",
            "hallucination_safety": "Hallucination Safety",
        }
        df_long = (
            df[["model_id"] + metric_columns]
            .melt(
                id_vars="model_id",
                value_vars=metric_columns,
                var_name="metric",
                value_name="score",
            )
            .assign(
                metric_label=lambda frame: frame["metric"].map(metric_title_map)
            )
        )
        metric_order = [metric_title_map[column] for column in metric_columns]

        chart_metrics = (
            alt.Chart(df_long)
            .mark_bar(size=40, cornerRadius=3)
            .encode(
                x=alt.X(
                    "model_id:N",
                    sort=alt.SortField(field="score", order="descending"),
                    title="Model",
                    axis=alt.Axis(labelAngle=-45, labelLimit=200),
                ),
                y=alt.Y(
                    "score:Q",
                    title="Average Score",
                    scale=alt.Scale(domain=[0, 5]),
                ),
                color=alt.Color(
                    "model_id:N",
                    scale=color_scale,
                    legend=alt.Legend(title="Model", columns=2),
                ),
                column=alt.Column(
                    "metric_label:N",
                    title=None,
                    sort=metric_order,
                    header=alt.Header(labelAngle=-90, labelAlign="left"),
                ),
                tooltip=[
                    alt.Tooltip("metric_label:N", title="Metric"),
                    alt.Tooltip("model_id:N", title="Model"),
                    alt.Tooltip("score:Q", title="Score", format=".2f"),
                ],
            )
            .properties(height=350, width=120)
            .resolve_scale(y="shared", x="independent")
        )
        st.altair_chart(chart_metrics, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Combined Metric Comparison")
        
        model_order = df["model_id"].tolist() if "overall_score" in df.columns else sorted(df["model_id"].unique())
        chart_combined = (
            alt.Chart(df_long)
            .mark_bar(size=35, cornerRadius=3)
            .encode(
                x=alt.X(
                    "metric_label:N",
                    title="Metric",
                    sort=metric_order,
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(
                    "score:Q",
                    title="Average Score",
                    scale=alt.Scale(domain=[0, 5]),
                ),
                color=alt.Color(
                    "model_id:N",
                    scale=color_scale,
                    legend=alt.Legend(title="Model", columns=2),
                ),
                column=alt.Column(
                    "model_id:N",
                    title="Model",
                    sort=model_order,
                    header=alt.Header(labelAngle=-90, labelAlign="left"),
                ),
                tooltip=[
                    alt.Tooltip("metric_label:N", title="Metric"),
                    alt.Tooltip("model_id:N", title="Model"),
                    alt.Tooltip("score:Q", title="Score", format=".2f"),
                ],
            )
            .properties(height=300, width=100)
            .resolve_scale(y="shared", x="independent")
        )
        st.altair_chart(chart_combined, use_container_width=True)


def render_domain_tab(summary: Dict[str, object]) -> None:
    st.subheader("Domain & Task Analysis")
    per_domain = summary.get("per_domain")
    per_task = summary.get("per_task_type")

    if per_domain:
        st.markdown("#### Performance by Domain")
        df_domain = pd.DataFrame(per_domain)
        if not df_domain.empty:
            st.dataframe(df_domain, use_container_width=True, hide_index=True)

            domains = sorted(df_domain["domain"].dropna().unique())
            if domains:
                selected_domain = st.selectbox(
                    "Select Domain for Visualization", domains, key="domain_viz_select"
                )
                domain_df = df_domain[df_domain["domain"] == selected_domain].copy()

                if not domain_df.empty:
                    metric_cols = [
                        col
                        for col in [
                            "correctness",
                            "methodology",
                            "specificity",
                            "clarity",
                            "hallucination_safety",
                        ]
                        if col in domain_df.columns
                    ]
                    if metric_cols:
                        domain_long = domain_df[["model_id"] + metric_cols].melt(
                            id_vars="model_id",
                            value_vars=metric_cols,
                            var_name="metric",
                            value_name="score",
                        )
                        metric_title_map = {
                            "correctness": "Correctness",
                            "methodology": "Methodology",
                            "specificity": "Specificity",
                            "clarity": "Clarity",
                            "hallucination_safety": "Hallucination Safety",
                        }
                        domain_long["metric_label"] = domain_long["metric"].map(metric_title_map)
                        metric_order = [metric_title_map[col] for col in metric_cols if col in metric_title_map]
                        
                        model_scores = domain_long.groupby("model_id")["score"].mean().sort_values(ascending=False)
                        model_order = model_scores.index.tolist()
                        
                        chart = (
                            alt.Chart(domain_long)
                            .mark_bar(size=40, cornerRadius=3)
                            .encode(
                                x=alt.X(
                                    "model_id:N",
                                    title="Model",
                                    sort=model_order,
                                    axis=alt.Axis(labelAngle=-45, labelLimit=200),
                                ),
                                y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 5])),
                                color=alt.Color("metric_label:N", legend=alt.Legend(title="Metric")),
                                column=alt.Column("metric_label:N", title=None, sort=metric_order),
                            )
                            .properties(height=300, width=150)
                            .resolve_scale(y="shared")
                        )
                        st.altair_chart(chart, use_container_width=True)

    if per_task:
        st.markdown("---")
        st.markdown("#### Performance by Task Type")
        df_task = pd.DataFrame(per_task)
        if not df_task.empty:
            st.dataframe(df_task, use_container_width=True, hide_index=True)

            task_types = sorted(df_task["task_type"].dropna().unique())
            if task_types:
                selected_task = st.selectbox(
                    "Select Task Type for Visualization",
                    task_types,
                    key="task_viz_select",
                )
                task_df = df_task[df_task["task_type"] == selected_task].copy()

                if not task_df.empty:
                    metric_cols = [
                        col
                        for col in [
                            "correctness",
                            "methodology",
                            "specificity",
                            "clarity",
                            "hallucination_safety",
                        ]
                        if col in task_df.columns
                    ]
                    if metric_cols:
                        task_long = task_df[["model_id"] + metric_cols].melt(
                            id_vars="model_id",
                            value_vars=metric_cols,
                            var_name="metric",
                            value_name="score",
                        )
                        metric_title_map = {
                            "correctness": "Correctness",
                            "methodology": "Methodology",
                            "specificity": "Specificity",
                            "clarity": "Clarity",
                            "hallucination_safety": "Hallucination Safety",
                        }
                        task_long["metric_label"] = task_long["metric"].map(metric_title_map)
                        metric_order = [metric_title_map[col] for col in metric_cols if col in metric_title_map]
                        
                        model_scores = task_long.groupby("model_id")["score"].mean().sort_values(ascending=False)
                        model_order = model_scores.index.tolist()
                        
                        chart = (
                            alt.Chart(task_long)
                            .mark_bar(size=40, cornerRadius=3)
                            .encode(
                                x=alt.X(
                                    "model_id:N",
                                    title="Model",
                                    sort=model_order,
                                    axis=alt.Axis(labelAngle=-45, labelLimit=200),
                                ),
                                y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 5])),
                                color=alt.Color("metric_label:N", legend=alt.Legend(title="Metric")),
                                column=alt.Column("metric_label:N", title=None, sort=metric_order),
                            )
                            .properties(height=300, width=150)
                            .resolve_scale(y="shared")
                        )
                        st.altair_chart(chart, use_container_width=True)


def render_analysis_tab(summary: Dict[str, object], insights: Dict[str, object]) -> None:
    st.subheader("Evaluation Analysis & Insights")
    
    if not insights:
        st.info("Run the analysis script to generate insights.json.")
        return
    
    per_model = summary.get("per_model", [])
    if not per_model:
        st.info("No model data available for analysis.")
        return
    
    df_models = pd.DataFrame(per_model)
    if "overall_score" in df_models.columns:
        df_models = df_models.sort_values(by="overall_score", ascending=False).reset_index(drop=True)
    
    truncation_stats = summary.get("truncation_stats", [])
    if truncation_stats:
        trunc_df = pd.DataFrame(truncation_stats)
        df_models = df_models.merge(
            trunc_df[["model_id", "truncation_rate"]],
            on="model_id",
            how="left"
        )
    
    st.markdown("### 📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Models", insights.get("total_models", 0))
    with col2:
        st.metric("Total Prompts", insights.get("total_prompts", 0))
    with col3:
        best_model = insights.get("best_overall_model", "N/A")
        best_score = insights.get("best_overall_score", 0)
        st.metric("Best Model", best_model.split("/")[-1] if "/" in best_model else best_model, f"{best_score:.2f}")
    with col4:
        trunc_rate = insights.get("truncation_summary", {}).get("truncation_rate_pct", 0)
        st.metric("Truncation Rate", f"{trunc_rate:.1f}%")
    
    st.markdown("---")
    st.markdown("### 🏆 Model Leaderboard")
    
    leaderboard_df = df_models.copy()
    leaderboard_df["rank"] = range(1, len(leaderboard_df) + 1)
    
    display_cols = ["rank", "model_id"]
    if "overall_score" in leaderboard_df.columns:
        display_cols.append("overall_score")
    metric_cols = ["correctness", "methodology", "specificity", "clarity", "hallucination_safety"]
    for col in metric_cols:
        if col in leaderboard_df.columns:
            display_cols.append(col)
    if "truncation_rate" in leaderboard_df.columns:
        display_cols.append("truncation_rate")
    
    leaderboard_display = leaderboard_df[[col for col in display_cols if col in leaderboard_df.columns]].copy()
    
    for col in leaderboard_display.columns:
        if col not in ["rank", "model_id"] and leaderboard_display[col].dtype in ["float64", "float32"]:
            leaderboard_display[col] = leaderboard_display[col].round(2)
    
    st.dataframe(leaderboard_display, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 📈 Key Insights")
    
    best_model = insights.get("best_overall_model", "N/A")
    best_score = insights.get("best_overall_score", 0)
    
    st.markdown(f"""
    #### Overall Performance
    - **Top Performing Model**: `{best_model}` achieved the highest overall score of **{best_score:.3f}/5.0**
    - **Evaluation Scale**: All models were evaluated on a 1-5 scale across 5 key metrics
    - **Total Evaluations**: {insights.get("total_prompts", 0)} prompt-response pairs across {insights.get("total_models", 0)} models
    """)
    
    top_models_by_metric = insights.get("top_models_by_metric", {})
    if top_models_by_metric:
        st.markdown("#### Top Models by Metric")
        metric_titles = {
            "correctness": "Correctness",
            "methodology": "Methodology",
            "specificity": "Specificity",
            "clarity": "Clarity",
            "hallucination_safety": "Hallucination Safety",
        }
        
        cols = st.columns(len(top_models_by_metric))
        for idx, (metric, data) in enumerate(top_models_by_metric.items()):
            with cols[idx]:
                model_name = data.get("model", "N/A")
                score = data.get("score", 0)
                st.metric(
                    metric_titles.get(metric, metric.replace("_", " ").title()),
                    model_name.split("/")[-1] if "/" in model_name else model_name,
                    f"{score:.2f}"
                )
    
    truncation_summary = insights.get("truncation_summary", {})
    if truncation_summary:
        total_truncated = truncation_summary.get("total_truncated", 0)
        trunc_rate = truncation_summary.get("truncation_rate_pct", 0)
        
        st.markdown("---")
        st.markdown("#### ⚠️ Data Quality Notes")
        st.info(
            f"**Truncation Analysis**: {total_truncated} responses ({trunc_rate:.1f}%) were truncated due to max_tokens limits. "
            "These responses may be incomplete. Models with higher truncation rates may have their scores affected. "
            "Truncation status is marked in the summary table and Q&A Explorer."
        )
    
    st.markdown("---")
    st.markdown("### 📋 Detailed Statistics")
    
    with st.expander("View Full Insights JSON", expanded=False):
        st.json(insights)
    
    with st.expander("View Model Performance Distribution", expanded=False):
        if "overall_score" in df_models.columns:
            st.markdown("**Overall Score Distribution**")
            score_stats = df_models["overall_score"].describe()
            st.dataframe(score_stats.to_frame().T, use_container_width=True)
            
            st.markdown("**Score Range Analysis**")
            col1, col2, col3 = st.columns(3)
            with col1:
                high_performers = len(df_models[df_models["overall_score"] >= 4.5])
                st.metric("High Performers (≥4.5)", high_performers)
            with col2:
                mid_performers = len(df_models[(df_models["overall_score"] >= 3.5) & (df_models["overall_score"] < 4.5)])
                st.metric("Mid Performers (3.5-4.5)", mid_performers)
            with col3:
                low_performers = len(df_models[df_models["overall_score"] < 3.5])
                st.metric("Lower Performers (<3.5)", low_performers)


def main() -> None:
    st.set_page_config(
        page_title="Scientific Evaluation Dashboard", layout="wide", initial_sidebar_state="expanded"
    )
    st.title("Scientific Evaluation Dashboard")
    st.caption(
        "Comprehensive evaluation results: explore Q&A pairs, model performance scores, "
        "and domain-specific analysis."
    )

    merged = _load_merged_results(MERGED_RESULTS_PATH)
    summary = _load_summary(SUMMARY_PATH)
    insights = _load_insights(INSIGHTS_PATH)

    truncation_count = merged["is_truncated"].sum() if "is_truncated" in merged.columns else 0
    total_count = len(merged)
    if truncation_count > 0:
        st.warning(
            f"⚠️ **Note**: {truncation_count} out of {total_count} responses ({100*truncation_count/total_count:.1f}%) "
            "were truncated due to max_tokens limits. These are marked in the Q&A Explorer."
        )

    tab_qna, tab_scores, tab_domain, tab_analysis = st.tabs(
        ["Q&A Explorer", "Performance Scores", "Domain Analysis", "Analysis & Insights"]
    )

    with tab_qna:
        render_qna_tab(merged)

    with tab_scores:
        render_scores_tab(summary)

    with tab_domain:
        render_domain_tab(summary)
    
    with tab_analysis:
        render_analysis_tab(summary, insights)


if __name__ == "__main__":
    main()
