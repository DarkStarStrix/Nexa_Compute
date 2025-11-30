# Nexa Evaluation Pipeline

> **Scope**: Model Grading, Analysis, and Benchmarking.
> **Modules**: `nexa_eval`

The Evaluation Pipeline assesses model performance using both deterministic metrics and LLM-as-a-Judge methodologies. It ensures that trained models meet quality standards before deployment.

## Core Components

### 1. Judge System (`nexa_eval/judge.py`)
Uses a strong teacher model (e.g., GPT-4o) to grade student model responses.
*   **Rubrics**: Configurable scoring criteria (accuracy, reasoning, formatting).
*   **Pairwise Comparison**: Compares two model outputs to determine a winner.

### 2. Analysis (`nexa_eval/analyze.py`)
Aggregates evaluation results into actionable reports.
*   **Win Rates**: Head-to-head performance against baselines.
*   **Error Analysis**: Clusters and categorizes failure modes.

## Usage

```bash
# Run evaluation on a distillation run
python nexa_eval/evaluate_distillation.py --run-id <run_id>
```

