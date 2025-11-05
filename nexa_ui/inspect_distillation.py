"""Streamlit dashboard for inspecting distillation data (teacher inputs/outputs)."""

import pandas as pd
import streamlit as st
from pathlib import Path
import re

# Set page config
st.set_page_config(
    page_title="Distillation Data Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@st.cache_data
def load_teacher_inputs(version: str = "v1", _mtime: float = 0.0) -> pd.DataFrame:
    """Load teacher inputs parquet."""
    path = PROJECT_ROOT / "data/processed/distillation/teacher_inputs" / f"teacher_inputs_{version}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_teacher_outputs(version: str = "v1", _mtime: float = 0.0) -> pd.DataFrame:
    """Load teacher outputs parquet."""
    path = PROJECT_ROOT / "data/processed/distillation/teacher_outputs" / f"teacher_outputs_{version}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def count_words(text: str) -> int:
    """Count words in text."""
    if pd.isna(text) or not text:
        return 0
    return len(re.findall(r'\b\w+\b', str(text)))


def merge_inputs_outputs(inputs_df: pd.DataFrame, outputs_df: pd.DataFrame) -> pd.DataFrame:
    """Merge inputs and outputs with deduplication.
    
    If outputs already have all required columns, use outputs directly.
    Otherwise, merge with inputs.
    """
    if len(outputs_df) == 0:
        return pd.DataFrame()
    
    required_cols = ["domain", "template_name", "user_prompt"]
    if "teacher_output" not in outputs_df.columns:
        return pd.DataFrame()
    
    # If outputs already have all required columns, use them directly
    if all(col in outputs_df.columns for col in required_cols):
        # Return all outputs - don't deduplicate since each row has unique teacher_output
        # Even if prompts are duplicates, the teacher responses are different
        return outputs_df.copy()
    
    # Otherwise, merge with inputs (legacy behavior)
    if len(inputs_df) == 0:
        return pd.DataFrame()
    
    if not all(col in inputs_df.columns for col in required_cols):
        return pd.DataFrame()
    
    # Create merge keys
    inputs_df = inputs_df.copy()
    outputs_df = outputs_df.copy()
    
    inputs_df["_merge_key"] = (
        inputs_df["domain"].astype(str) + "|||" + 
        inputs_df["template_name"].astype(str) + "|||" + 
        inputs_df["user_prompt"].astype(str)
    )
    
    outputs_df["_merge_key"] = (
        outputs_df["domain"].astype(str) + "|||" + 
        outputs_df["template_name"].astype(str) + "|||" + 
        outputs_df["user_prompt"].astype(str)
    )
    
    # Deduplicate outputs (keep first occurrence)
    outputs_df = outputs_df.drop_duplicates(subset=["_merge_key"], keep="first")
    
    # Merge
    combined = inputs_df.merge(
        outputs_df[["_merge_key", "teacher_output", "model_id", "latency_ms", "total_tokens", "prompt_tokens", "completion_tokens"]],
        on="_merge_key",
        how="left",
        suffixes=("", "_output")
    )
    
    # Drop merge key
    combined = combined.drop(columns=["_merge_key"])
    
    return combined


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute statistics on the dataset."""
    stats = {}
    
    if len(df) == 0:
        return stats
    
    # Domain distribution
    if "domain" in df.columns:
        stats["domain_dist"] = df["domain"].value_counts().to_dict()
    
    # Template distribution
    if "template_name" in df.columns:
        stats["template_dist"] = df["template_name"].value_counts().to_dict()
    
    # Domain/template combinations
    if "domain" in df.columns and "template_name" in df.columns:
        stats["domain_template_dist"] = df.groupby(["domain", "template_name"]).size().to_dict()
    
    # Word counts
    if "user_prompt" in df.columns:
        df = df.copy()
        df["_prompt_words"] = df["user_prompt"].apply(count_words)
        stats["prompt_word_stats"] = {
            "mean": float(df["_prompt_words"].mean()),
            "median": float(df["_prompt_words"].median()),
            "min": int(df["_prompt_words"].min()),
            "max": int(df["_prompt_words"].max()),
        }
    
    if "teacher_output" in df.columns:
        df = df.copy()
        df["_output_words"] = df["teacher_output"].apply(lambda x: count_words(x) if pd.notna(x) else 0)
        stats["output_word_stats"] = {
            "mean": float(df[df["_output_words"] > 0]["_output_words"].mean()) if (df["_output_words"] > 0).any() else 0.0,
            "median": float(df[df["_output_words"] > 0]["_output_words"].median()) if (df["_output_words"] > 0).any() else 0.0,
            "min": int(df[df["_output_words"] > 0]["_output_words"].min()) if (df["_output_words"] > 0).any() else 0,
            "max": int(df[df["_output_words"] > 0]["_output_words"].max()) if (df["_output_words"] > 0).any() else 0,
        }
        stats["outputs_with_content"] = int((df["_output_words"] > 0).sum())
    
    # Latency stats
    if "latency_ms" in df.columns:
        df = df.copy()
        df["_latency"] = pd.to_numeric(df["latency_ms"], errors="coerce")
        stats["latency_stats"] = {
            "mean": float(df["_latency"].mean()),
            "median": float(df["_latency"].median()),
            "min": float(df["_latency"].min()),
            "max": float(df["_latency"].max()),
        }
    
    # Token stats
    if "total_tokens" in df.columns:
        df = df.copy()
        df["_tokens"] = pd.to_numeric(df["total_tokens"], errors="coerce")
        stats["token_stats"] = {
            "mean": float(df["_tokens"].mean()),
            "median": float(df["_tokens"].median()),
            "min": int(df["_tokens"].min()),
            "max": int(df["_tokens"].max()),
        }
    
    return stats


def main():
    """Main dashboard."""
    st.title("🔬 Distillation Data Inspector")
    st.markdown("Inspect teacher inputs and outputs for knowledge distillation pipeline")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        version = st.selectbox("Version", ["v1"], index=0)
        
        # Clear cache button
        if st.button("🔄 Reload Data", help="Clear cache and reload data from files"):
            load_teacher_inputs.clear()
            load_teacher_outputs.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Load data
        inputs_path = PROJECT_ROOT / "data/processed/distillation/teacher_inputs" / f"teacher_inputs_{version}.parquet"
        outputs_path = PROJECT_ROOT / "data/processed/distillation/teacher_outputs" / f"teacher_outputs_{version}.parquet"
        inputs_mtime = inputs_path.stat().st_mtime if inputs_path.exists() else 0.0
        outputs_mtime = outputs_path.stat().st_mtime if outputs_path.exists() else 0.0
        
        inputs_df = load_teacher_inputs(version, _mtime=inputs_mtime)
        outputs_df = load_teacher_outputs(version, _mtime=outputs_mtime)
        
        st.metric("Teacher Inputs", len(inputs_df))
        st.metric("Teacher Outputs", len(outputs_df))
        
        if len(inputs_df) > 0 and len(outputs_df) > 0:
            completion_rate = (len(outputs_df) / len(inputs_df)) * 100
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    # Merge data
    combined_df = merge_inputs_outputs(inputs_df, outputs_df)
    
    if len(combined_df) == 0:
        st.warning("No data found. Run the data generation job first.")
        return
    
    # Filter to only rows with outputs for main view
    qa_df = combined_df[combined_df["teacher_output"].notna() & (combined_df["teacher_output"] != "")].copy()
    
    if len(qa_df) == 0:
        st.warning("No QA pairs found with teacher outputs.")
        return
    
    # Statistics
    stats = compute_statistics(qa_df)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📋 QA Pairs", "📊 Statistics & Distribution"])
    
    with tab1:
        st.header("QA Pairs")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            available_domains = sorted(qa_df["domain"].unique().tolist()) if "domain" in qa_df.columns else []
            domain_filter = st.multiselect(
                "Filter by Domain",
                available_domains,
                key="qa_domain_filter",
            )
        with col2:
            available_templates = sorted(qa_df["template_name"].unique().tolist()) if "template_name" in qa_df.columns else []
            template_filter = st.multiselect(
                "Filter by Template",
                available_templates,
                key="qa_template_filter",
            )
        with col3:
            st.write("")  # Spacing
            show_stats = st.checkbox("Show sample stats", key="show_sample_stats", value=False)
        
        # Apply filters and reset index
        filtered_df = qa_df.copy()
        if domain_filter:
            filtered_df = filtered_df[filtered_df["domain"].isin(domain_filter)]
        if template_filter:
            filtered_df = filtered_df[filtered_df["template_name"].isin(template_filter)]
        
        # Reset index to ensure iloc works correctly
        filtered_df = filtered_df.reset_index(drop=True)
        
        st.caption(f"Showing {len(filtered_df)} of {len(qa_df)} QA pairs")
        
        if len(filtered_df) == 0:
            st.info("No QA pairs match the selected filters.")
            return
        
        # Create a unique key based on filter state - this forces widget reset when filters change
        domain_key = ",".join(sorted(domain_filter or []))
        template_key = ",".join(sorted(template_filter or []))
        filter_key_str = f"d_{domain_key}_t_{template_key}"
        selector_key = f"qa_selector_{filter_key_str}"
        
        # Initialize session state for this filter combination if it doesn't exist
        if selector_key not in st.session_state:
            st.session_state[selector_key] = 0
        
        # QA pair selector - key includes filter state so it resets when filters change
        # The widget automatically updates session state with its key
        row_idx = st.number_input(
            "Select QA Pair",
            min_value=0,
            max_value=max(0, len(filtered_df) - 1),
            value=st.session_state[selector_key],
            step=1,
            key=selector_key,
            help="Navigate through QA pairs (use +/- buttons or type a number)"
        )
        
        # Ensure row_idx matches the widget value (it should, but be explicit)
        row_idx = int(st.session_state[selector_key])
        
        # Ensure row_idx is within bounds
        row_idx = min(max(0, row_idx), len(filtered_df) - 1)
        
        # Get the selected row - reset index ensures iloc works correctly
        if len(filtered_df) > 0 and 0 <= row_idx < len(filtered_df):
            selected_row = filtered_df.iloc[row_idx].copy()
        else:
            # Fallback to first row if something goes wrong
            row_idx = 0
            selected_row = filtered_df.iloc[0].copy()
        
        # Create unique keys for this row to prevent caching issues
        row_key = f"{filter_key_str}_{row_idx}"
        
        # Display QA pair
        st.markdown("---")
        
        # Sample counter and metadata header with debug info
        sample_info = f"📄 **Sample {row_idx + 1} of {len(filtered_df)}**"
        if len(filtered_df) > 0:
            # Show a preview of the prompt to verify it's changing
            prompt_preview = str(selected_row.get('user_prompt', ''))[:50].replace('\n', ' ')
            sample_info += f" | Domain: {selected_row.get('domain', 'N/A')}"
            with st.expander("🔍 Debug: Verify row is changing", expanded=False):
                st.write(f"**Row Index:** {row_idx}")
                st.write(f"**Prompt Preview:** {prompt_preview}...")
                st.write(f"**Full Row Hash:** {hash(str(selected_row.to_dict()))}")
        st.caption(sample_info)
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        with meta_col1:
            st.caption(f"**Domain:** {selected_row.get('domain', 'N/A')}")
        with meta_col2:
            st.caption(f"**Template:** {selected_row.get('template_name', 'N/A')}")
        with meta_col3:
            if show_stats and "latency_ms" in selected_row:
                st.caption(f"**Latency:** {selected_row['latency_ms']:.0f} ms")
        with meta_col4:
            if show_stats and "total_tokens" in selected_row:
                st.caption(f"**Tokens:** {selected_row['total_tokens']}")
        
        st.markdown("---")
        
        # Question section
        st.markdown("### ❓ Question")
        question_text = str(selected_row.get("user_prompt", ""))
        st.text_area(
            "User Prompt",
            question_text,
            height=150,
            key=f"qa_question_{row_key}",
            label_visibility="collapsed",
            disabled=True,
        )
        
        # System prompt section
        if "system_prompt" in selected_row and pd.notna(selected_row["system_prompt"]):
            with st.expander("📝 System Prompt", expanded=False):
                st.text_area(
                    "System Prompt",
                    str(selected_row["system_prompt"]),
                    height=200,
                    key=f"qa_system_prompt_{row_key}",
                    label_visibility="collapsed",
                    disabled=True,
                )
        
        # Answer section
        st.markdown("### 💡 Answer")
        answer_text = str(selected_row.get("teacher_output", ""))
        st.text_area(
            "Teacher Output",
            answer_text,
            height=400,
            key=f"qa_answer_{row_key}",
            label_visibility="collapsed",
            disabled=True,
        )
        
        # Full details expander
        with st.expander("🔍 Full Details", expanded=False):
            detail_cols = [col for col in selected_row.index if col not in ["user_prompt", "teacher_output", "system_prompt"]]
            details = {col: selected_row[col] for col in detail_cols}
            st.json(details)
    
    with tab2:
        st.header("Statistics & Distribution")
        
        # Domain distribution
        if "domain_dist" in stats:
            st.markdown("### Domain Distribution")
            domain_dist = stats["domain_dist"]
            dist_df = pd.DataFrame(list(domain_dist.items()), columns=["Domain", "Count"])
            dist_df = dist_df.sort_values("Count", ascending=False)
            st.bar_chart(dist_df.set_index("Domain"))
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Template distribution
        if "template_dist" in stats:
            st.markdown("### Template Distribution")
            template_dist = stats["template_dist"]
            dist_df = pd.DataFrame(list(template_dist.items()), columns=["Template", "Count"])
            dist_df = dist_df.sort_values("Count", ascending=False)
            st.bar_chart(dist_df.set_index("Template"))
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Domain/Template combinations
        if "domain_template_dist" in stats:
            st.markdown("### Domain × Template Distribution")
            combo_dist = stats["domain_template_dist"]
            combo_list = [{"Domain": k[0], "Template": k[1], "Count": v} for k, v in combo_dist.items()]
            combo_df = pd.DataFrame(combo_list)
            combo_df = combo_df.sort_values(["Domain", "Template"])
            st.dataframe(combo_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Word count statistics
        col1, col2 = st.columns(2)
        
        with col1:
            if "prompt_word_stats" in stats:
                st.markdown("### Prompt Word Counts")
                prompt_stats = stats["prompt_word_stats"]
                st.metric("Mean", f"{prompt_stats['mean']:.1f}")
                st.metric("Median", f"{prompt_stats['median']:.1f}")
                st.metric("Min", prompt_stats['min'])
                st.metric("Max", prompt_stats['max'])
        
        with col2:
            if "output_word_stats" in stats:
                st.markdown("### Output Word Counts")
                output_stats = stats["output_word_stats"]
                st.metric("Mean", f"{output_stats['mean']:.1f}")
                st.metric("Median", f"{output_stats['median']:.1f}")
                st.metric("Min", output_stats['min'])
                st.metric("Max", output_stats['max'])
                if "outputs_with_content" in stats:
                    st.metric("With Content", stats['outputs_with_content'])
        
        st.markdown("---")
        
        # Performance statistics
        col1, col2 = st.columns(2)
        
        with col1:
            if "latency_stats" in stats:
                st.markdown("### Latency (ms)")
                latency_stats = stats["latency_stats"]
                st.metric("Mean", f"{latency_stats['mean']:.1f}")
                st.metric("Median", f"{latency_stats['median']:.1f}")
                st.metric("Min", f"{latency_stats['min']:.1f}")
                st.metric("Max", f"{latency_stats['max']:.1f}")
        
        with col2:
            if "token_stats" in stats:
                st.markdown("### Token Usage")
                token_stats = stats["token_stats"]
                st.metric("Mean", f"{token_stats['mean']:.1f}")
                st.metric("Median", f"{token_stats['median']:.1f}")
                st.metric("Min", token_stats['min'])
                st.metric("Max", token_stats['max'])


if __name__ == "__main__":
    main()

