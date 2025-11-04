"""Streamlit UI for manual inspection of teacher outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from .utils import read_parquet, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments passed through Streamlit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Parquet file to inspect")
    parser.add_argument(
        "--label-path",
        type=Path,
        default=Path("/mnt/nexa_durable/distill/labels")
        / f"labels_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    parser.add_argument("--start-index", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args


def load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset into DataFrame with sensible defaults."""

    df = read_parquet(path)
    expected_columns = {"context", "teacher_output", "task_type"}
    missing = expected_columns - set(df.columns)
    if missing:
        st.warning(f"Missing expected columns: {missing}")
    return df


def init_state(total_rows: int, start_index: int) -> None:
    """Initialize Streamlit session state."""

    if "row_index" not in st.session_state:
        st.session_state.row_index = max(0, min(total_rows - 1, start_index))
    if "annotations" not in st.session_state:
        st.session_state.annotations: List[dict] = []


def record_annotation(action: str, row: pd.Series) -> None:
    """Record a label for the current row."""

    annotation = {
        "index": int(row.name),
        "task_type": row.get("task_type"),
        "context": row.get("context"),
        "teacher_output": row.get("teacher_output"),
        "action": action,
        "timestamp": datetime.utcnow().isoformat(),
    }
    st.session_state.annotations.append(annotation)


def navigate(delta: int, total_rows: int) -> None:
    """Advance or rewind the current row index."""

    st.session_state.row_index = (st.session_state.row_index + delta) % max(total_rows, 1)


def main() -> None:
    """Render the Streamlit inspection UI."""

    args = parse_args()
    st.set_page_config(page_title="Nexa Distill Inspector", layout="wide")
    st.title("Nexa Distill — Manual Inspection")
    st.caption(f"Dataset: {args.src}")

    df = load_dataset(args.src)
    if df.empty:
        st.error("Dataset is empty")
        return

    init_state(len(df), args.start_index)
    row = df.iloc[st.session_state.row_index]

    st.sidebar.header("Navigation")
    st.sidebar.write(f"Total rows: {len(df)}")
    st.sidebar.write(f"Current index: {st.session_state.row_index}")
    if st.sidebar.button("Prev", use_container_width=True):
        navigate(-1, len(df))
    if st.sidebar.button("Next", use_container_width=True):
        navigate(1, len(df))

    st.sidebar.header("Annotations")
    st.sidebar.write(f"Recorded: {len(st.session_state.annotations)}")

    st.subheader("Context")
    st.text_area("Scientific Context", value=row.get("context", ""), height=300)

    st.subheader("Teacher Output")
    st.text_area("Teacher Output", value=row.get("teacher_output", ""), height=300)

    st.subheader("Metadata")
    st.json({col: row[col] for col in df.columns if col not in {"context", "teacher_output"}}, expanded=False)

    col_accept, col_regen, col_reject = st.columns(3)
    if col_accept.button("Accept", use_container_width=True):
        record_annotation("accept", row)
        navigate(1, len(df))
    if col_regen.button("Regenerate", use_container_width=True):
        record_annotation("regenerate", row)
        navigate(1, len(df))
    if col_reject.button("Reject", use_container_width=True):
        record_annotation("reject", row)
        navigate(1, len(df))

    st.sidebar.header("Persistence")
    if st.sidebar.button("Save annotations", use_container_width=True):
        if st.session_state.annotations:
            write_jsonl(st.session_state.annotations, args.label_path)
            st.success(f"Saved {len(st.session_state.annotations)} annotations to {args.label_path}")
        else:
            st.info("No annotations to save.")


if __name__ == "__main__":  # pragma: no cover - Streamlit entry
    main()

