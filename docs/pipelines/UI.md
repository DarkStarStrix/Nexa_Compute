# Nexa User Interface

> **Scope**: Dashboards and Visualization.
> **Modules**: `nexa_ui`

The `nexa_ui` module provides a comprehensive suite of web-based interfaces for monitoring the platform's activities, from data ingestion to model evaluation. These dashboards are built using **Streamlit** for interactivity and ease of deployment.

## Dashboards

### 1. MS/MS Data Inspector (`nexa_ui/msms_dashboard.py`)

**Purpose**: Real-time monitoring and deep inspection of the Scientific Data Refinery (MS/MS pipeline). It bridges the gap between raw binary Parquet shards and human-readable spectral data.

**Key Features**:
*   **Overview Tab**: Displays global dataset metrics (Total Shards, Total Samples, File Size) and the latest `quality_report.json` status (PASS/WARN/FAIL).
*   **Structural Integrity**: Visualizes error rates, validation failures (e.g., NaN values, shape mismatches), and duplicate detection results.
*   **Semantic Quality**:
    *   **Peak Count Histograms**: Analyzes the distribution of peaks per spectrum.
    *   **Precursor m/z Distribution**: Verifies mass range coverage.
    *   **Adduct & Charge State**: Pie/Bar charts showing the distribution of ionization states.
*   **Samples Inspector**: An interactive "Spectrum Viewer" that plots m/z vs. Intensity for individual samples, along with their metadata (SMILES, InChIKey, Instrument).
*   **Pipeline Health**: Tracks throughput (spectra/sec) and resource utilization metrics.

**Usage**:
```bash
# Run as a full dashboard
streamlit run nexa_ui/msms_dashboard.py -- --ui

# Run in CLI mode for quick stats
python -m nexa_ui.msms_dashboard stats --dataset gems_full
```

### 2. Evaluation Dashboard (`nexa_ui/eval_dashboard.py`)

**Purpose**: A centralized hub for analyzing model performance benchmarks and "LLM-as-a-Judge" results.

**Key Features**:
*   **Q&A Explorer**: A detailed view of individual prompt-response pairs. Users can filter by domain (e.g., "Physics", "Biology") and see the exact output alongside judge scores (Correctness, Methodology, etc.).
*   **Performance Scores**: Aggregated tables showing mean scores per model. Includes truncation analysis to identify models hitting token limits.
*   **Domain Analysis**: Heatmaps and bar charts breaking down performance by scientific domain and task type (e.g., "Hypothesis Generation" vs. "Methodology").
*   **Leaderboard**: A ranked view of all models based on weighted average scores, highlighting top performers and "win rates".
*   **Insights**: Auto-generated executive summaries identifying the best model, common failure modes, and data quality issues.

### 3. Distillation Inspector (`nexa_ui/inspect_distillation.py`)

**Purpose**: A quality control tool for the Knowledge Distillation pipeline. It allows engineers to validate the quality of teacher-generated data before it enters the training set.

**Key Features**:
*   **QA Pairs View**: Side-by-side display of User Prompts (Teacher Inputs) and Teacher Outputs.
*   **Statistics & Distribution**:
    *   **Word Counts**: Histograms of prompt and response lengths.
    *   **Latency**: Analysis of generation time (ms) to detect performance regressions.
    *   **Token Usage**: Cost estimation based on token counts.
*   **Filtering**: Interactive widgets to slice data by Domain or Template Name, helping identify under-represented categories.

### 4. vLLM Benchmark Dashboard (`nexa_ui/vllm_dashboard.py`)

**Purpose**: Performance profiling for the Inference Engine.

**Key Features**:
*   **Throughput Analysis**: Visualizes Tokens/Second and Requests/Second across different hardware configurations (e.g., TP=1 vs TP=2).
*   **Latency Breakdown**: Analyzes Time-To-First-Token (TTFT) and total generation time.
*   **Config Explorer**: Detailed view of engine parameters (`gpu_memory_utilization`, `max_model_len`) used for each run.

### 5. General Leaderboard (`nexa_ui/leaderboard.py`)

**Purpose**: High-level project tracking.

**Features**:
*   **Evaluation Tracker**: Comparison of different training runs against baselines.
*   **Data Volume**: Tracks the growth of processed datasets over time.

## BI Integrations

While Streamlit serves operational needs, NexaCompute aims to integrate with enterprise BI tools for long-term trending:
*   **Superset/Metabase**: Future integrations will consume the `metrics.json` and `quality_report.json` artifacts produced by pipelines to build historical trend dashboards.

## deployment

Dashboards are typically deployed as standalone services within the cluster or on a local dev machine.

**Docker**:
The `nexa_infra` container registry includes dependencies for all UI tools.

```bash
# Launch from within the nexa_tools container
python orchestrate.py run ui
```
