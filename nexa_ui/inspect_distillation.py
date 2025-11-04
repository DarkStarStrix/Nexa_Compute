"""Streamlit dashboard for inspecting distillation data (teacher inputs/outputs)."""

import pandas as pd
import streamlit as st
from pathlib import Path

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
def load_teacher_inputs(version: str = "v1") -> pd.DataFrame:
    """Load teacher inputs parquet."""
    path = PROJECT_ROOT / "data/processed/distillation/teacher_inputs" / f"teacher_inputs_{version}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_teacher_outputs(version: str = "v1") -> pd.DataFrame:
    """Load teacher outputs parquet."""
    path = PROJECT_ROOT / "data/processed/distillation/teacher_outputs" / f"teacher_outputs_{version}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main():
    """Main dashboard."""
    st.title("🔬 Distillation Data Inspector")
    st.markdown("Inspect teacher inputs and outputs for knowledge distillation pipeline")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        version = st.selectbox("Version", ["v1"], index=0)
        st.markdown("---")
        
        # Quick stats
        inputs_df = load_teacher_inputs(version)
        outputs_df = load_teacher_outputs(version)
        
        st.metric("Teacher Inputs", len(inputs_df))
        st.metric("Teacher Outputs", len(outputs_df))
        
        if len(inputs_df) > 0 and len(outputs_df) > 0:
            completion_rate = (len(outputs_df) / len(inputs_df)) * 100
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Teacher Inputs", "📤 Teacher Outputs", "🔗 Combined View", "📊 Statistics"])
    
    with tab1:
        st.header("Teacher Inputs")
        if len(inputs_df) > 0:
            st.dataframe(inputs_df, use_container_width=True, height=400)
            
            # Column info
            if st.checkbox("Show Column Info", key="inputs_info"):
                st.json(inputs_df.dtypes.to_dict())
            
            # Sample row
            if st.checkbox("Show Sample Row", key="inputs_sample"):
                if len(inputs_df) > 0:
                    sample_idx = st.number_input("Row Index", 0, len(inputs_df)-1, 0, key="inputs_idx")
                    st.json(inputs_df.iloc[sample_idx].to_dict())
        else:
            st.warning("No teacher inputs found. Run the data generation job first.")
    
    with tab2:
        st.header("Teacher Outputs")
        if len(outputs_df) > 0:
            st.dataframe(outputs_df, use_container_width=True, height=400)
            
            # Column info
            if st.checkbox("Show Column Info", key="outputs_info"):
                st.json(outputs_df.dtypes.to_dict())
            
            # Sample output
            if st.checkbox("Show Sample Output", key="outputs_sample"):
                if len(outputs_df) > 0:
                    sample_idx = st.number_input("Row Index", 0, len(outputs_df)-1, 0, key="outputs_idx")
                    row = outputs_df.iloc[sample_idx]
                    st.markdown("### Teacher Output Text")
                    st.text_area("Output", row.get("teacher_output", ""), height=200, key="output_text")
                    if "latency_ms" in row:
                        st.metric("Latency", f"{row['latency_ms']} ms")
                    if "total_tokens" in row:
                        st.metric("Total Tokens", row["total_tokens"])
        else:
            st.warning("No teacher outputs found. Run the data generation job first.")
    
    with tab3:
        st.header("Combined View")
        if len(inputs_df) > 0 and len(outputs_df) > 0:
            # Merge inputs and outputs
            if "user_prompt" in inputs_df.columns and "teacher_output" in outputs_df.columns:
                combined = inputs_df.merge(
                    outputs_df[["teacher_output", "model_id", "latency_ms", "total_tokens"]],
                    left_index=True,
                    right_index=True,
                    how="left"
                )
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    domain_filter = st.multiselect(
                        "Filter by Domain",
                        combined["domain"].unique() if "domain" in combined.columns else [],
                        key="domain_filter"
                    )
                with col2:
                    template_filter = st.multiselect(
                        "Filter by Template",
                        combined["template_name"].unique() if "template_name" in combined.columns else [],
                        key="template_filter"
                    )
                
                # Apply filters
                filtered = combined.copy()
                if domain_filter:
                    filtered = filtered[filtered["domain"].isin(domain_filter)]
                if template_filter:
                    filtered = filtered[filtered["template_name"].isin(template_filter)]
                
                st.dataframe(filtered, use_container_width=True, height=400)
                
                # Show QA pair
                if st.checkbox("Show QA Pair", key="qa_pair"):
                    qa_idx = st.number_input("Row Index", 0, len(filtered)-1, 0, key="qa_idx")
                    row = filtered.iloc[qa_idx]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Input (Prompt)")
                        st.text_area("User Prompt", row.get("user_prompt", ""), height=150, key="qa_prompt")
                        if "system_prompt" in row:
                            st.text_area("System Prompt", row.get("system_prompt", ""), height=100, key="qa_system")
                    
                    with col2:
                        st.markdown("### Output (Teacher Response)")
                        st.text_area("Teacher Output", row.get("teacher_output", ""), height=250, key="qa_output")
                        if "latency_ms" in row:
                            st.caption(f"Latency: {row['latency_ms']} ms | Tokens: {row.get('total_tokens', 'N/A')}")
            else:
                st.warning("Cannot merge - missing required columns")
        else:
            st.warning("Need both inputs and outputs to show combined view")
    
    with tab4:
        st.header("Statistics & Analysis")
        
        if len(inputs_df) > 0:
            st.subheader("Input Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "domain" in inputs_df.columns:
                    st.markdown("### Domain Distribution")
                    domain_counts = inputs_df["domain"].value_counts()
                    st.bar_chart(domain_counts)
            
            with col2:
                if "template_name" in inputs_df.columns:
                    st.markdown("### Template Distribution")
                    template_counts = inputs_df["template_name"].value_counts()
                    st.bar_chart(template_counts)
            
            with col3:
                if "user_prompt" in inputs_df.columns:
                    st.markdown("### Prompt Length")
                    prompt_lengths = inputs_df["user_prompt"].str.len()
                    st.metric("Avg Length", f"{prompt_lengths.mean():.0f} chars")
                    st.metric("Min Length", f"{prompt_lengths.min():.0f} chars")
                    st.metric("Max Length", f"{prompt_lengths.max():.0f} chars")
        
        if len(outputs_df) > 0:
            st.subheader("Output Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "latency_ms" in outputs_df.columns:
                    st.markdown("### Latency")
                    st.metric("Avg Latency", f"{outputs_df['latency_ms'].mean():.0f} ms")
                    st.metric("Min Latency", f"{outputs_df['latency_ms'].min():.0f} ms")
                    st.metric("Max Latency", f"{outputs_df['latency_ms'].max():.0f} ms")
            
            with col2:
                if "total_tokens" in outputs_df.columns:
                    st.markdown("### Token Usage")
                    st.metric("Total Tokens", f"{outputs_df['total_tokens'].sum():,}")
                    st.metric("Avg Tokens", f"{outputs_df['total_tokens'].mean():.0f}")
                    st.metric("Max Tokens", f"{outputs_df['total_tokens'].max():,}")
            
            with col3:
                if "teacher_output" in outputs_df.columns:
                    st.markdown("### Output Length")
                    output_lengths = outputs_df["teacher_output"].str.len()
                    st.metric("Avg Length", f"{output_lengths.mean():.0f} chars")
                    st.metric("Min Length", f"{output_lengths.min():.0f} chars")
                    st.metric("Max Length", f"{output_lengths.max():.0f} chars")
            
            # Token breakdown
            if "prompt_tokens" in outputs_df.columns and "completion_tokens" in outputs_df.columns:
                st.markdown("### Token Breakdown")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Prompt Tokens", f"{outputs_df['prompt_tokens'].sum():,}")
                    st.metric("Avg Prompt Tokens", f"{outputs_df['prompt_tokens'].mean():.0f}")
                with col2:
                    st.metric("Total Completion Tokens", f"{outputs_df['completion_tokens'].sum():,}")
                    st.metric("Avg Completion Tokens", f"{outputs_df['completion_tokens'].mean():.0f}")


if __name__ == "__main__":
    main()

