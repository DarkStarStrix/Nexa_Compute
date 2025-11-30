# Nexa User Interface

> **Scope**: Dashboards and Visualization.
> **Modules**: `nexa_ui`

The UI module provides web-based interfaces for monitoring the platform's activities.

## Dashboards

### 1. MS/MS Dashboard (`nexa_ui/msms_dashboard.py`)
*   **Purpose**: Real-time monitoring of the Scientific Data Refinery.
*   **Metrics**: Processing throughput, attrition rates, data quality distributions.

### 2. Evaluation Dashboard (`nexa_ui/eval_dashboard.py`)
*   **Purpose**: Visualizing model benchmarks and judge results.
*   **Features**: Win-rate matrices, sample inspection, error clustering.

### 3. Distillation Inspector (`nexa_ui/inspect_distillation.py`)
*   **Purpose**: Reviewing teacher-student data pairs.
*   **Features**: Quality filtering, prompt refinement UI.

## BI Dashboards

Placeholder for BI dashboard configs (e.g., Superset, Metabase) consumed by the Nexa UI layer.

## Usage

Dashboards are typically launched via `streamlit` or `fastapi`:

```bash
streamlit run nexa_ui/msms_dashboard.py
```
