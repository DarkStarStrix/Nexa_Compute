# Nexa UI

> 📚 **Full Documentation**: [docs/pipelines/UI.md](../../docs/pipelines/UI.md)

## Overview

The `nexa_ui` module provides a suite of interactive dashboards for visualizing the status and results of the Nexa Compute pipelines. Built on **Streamlit**, these tools allow researchers and engineers to inspect data quality, monitor training progress, benchmark inference performance, and analyze evaluation metrics without writing code.

## Key Components

### `eval_dashboard.py`
A comprehensive dashboard for analyzing LLM evaluation results.

#### Features
*   **Q&A Explorer:** Browse individual prompt-response pairs with associated judge scores.
*   **Performance Scores:** Aggregate metrics (Correctness, Methodology, etc.) per model.
*   **Domain Analysis:** Drill down into model performance by scientific domain or task type.
*   **Insights:** Auto-generated executive summaries and leaderboards.

### `inspect_distillation.py`
A specialized tool for validating the quality of teacher-generated datasets before they are used for training.

#### Features
*   **Data Inspection:** View raw teacher inputs (prompts) and outputs.
*   **Statistics:** Analyze word counts, token usage, and latency distributions.
*   **Filtering:** Slice data by domain and template type to identify gaps or issues.

### `leaderboard.py`
A high-level comparison tool for tracking model progress.

#### Features
*   **Evaluation Leaderboard:** Compare different training runs against standard benchmarks.
*   **Data Tracking:** Visualize the growth of distillation and training datasets over time.

### `msms_dashboard.py`
An inspector for Mass Spectrometry (MS/MS) data shards, bridging the gap between raw binary data and human readability.

#### Features
*   **Spectrum Viewer:** Interactive plots of mass spectra (m/z vs intensity).
*   **Quality Reports:** Visualize structural integrity errors and attrition rates.
*   **Shard Inspection:** Browse metadata, schema, and samples within Parquet shards.
*   **CLI Integration:** Can be run as a command-line tool for quick stats (`python -m nexa_ui.msms_dashboard stats`).

### `vllm_dashboard.py`
A benchmarking dashboard for analyzing inference server performance.

#### Features
*   **Throughput Analysis:** Visualize tokens/sec and requests/sec across different configurations.
*   **Latency Breakdown:** Identify bottlenecks in time-to-first-token (TTFT) and total generation time.
*   **Config Explorer:** Compare performance impacts of parameters like `tensor_parallel_size` and quantization.
