# Nexa Eval

> 📚 **Full Documentation**: [docs/pipelines/EVALUATION.md](../../docs/pipelines/EVALUATION.md)

## Overview

The `nexa_eval` module provides a comprehensive evaluation framework for Large Language Models (LLMs). It supports:

*   **LLM-as-a-Judge:** Automated scoring of model outputs using capable judge models (e.g., GPT-4).
*   **Distillation Evaluation:** assessing the quality and diversity of teacher-generated datasets.
*   **Generation Benchmarking:** running inference sweeps across local and remote models.
*   **Tool Use Evaluation:** specific metrics for verifying tool calls, argument validity, and citation accuracy in agentic workflows.

## Key Components

### `analyze_results.py`
Merges generation outputs and judgment scores to produce final evaluation reports and visualizations.

#### Functions
*   `merge_data(data: EvaluationData) -> pd.DataFrame`
    *   Joins prompts, model outputs, and judge scores into a single DataFrame for analysis.
*   `compute_metrics(merged: pd.DataFrame) -> Dict[str, object]`
    *   Calculates aggregate scores per model, domain, and task type.

### `evaluate_distillation.py`
Evaluates the raw materials of the distillation process—teacher inputs and outputs—before they are used for training.

#### Functions
*   `evaluate_teacher_outputs(outputs_path: Path) -> Dict[str, float]`
    *   Computes metrics like average length, completion rate, latency, and cost estimates for teacher responses.
*   `evaluate_teacher_inputs(inputs_path: Path) -> Dict[str, float]`
    *   Analyzes the distribution and quality of prompts used to query the teacher.

### `generate_responses.py`
The main driver for running inference benchmarks. It supports both local models (via Nexa's FastAPI server) and remote models (via OpenRouter).

#### Functions
*   `generate_for_local(...)`
    *   Sends batch requests to a locally running inference server.
*   `generate_for_openrouter(...)`
    *   Manages concurrent requests to OpenRouter APIs with retry logic and cost tracking.
*   `build_records(...)`
    *   Combines raw model responses with metadata for downstream judging.

### `judge_responses.py`
Implements the LLM-as-a-Judge workflow, scoring model outputs against defined rubrics.

#### Functions
*   `judge_model_outputs(...) -> pd.DataFrame`
    *   Orchestrates the judging process: constructs prompts for the judge, calls the API, and parses the JSON scores.
*   `parse_judge_response(payload: str)`
    *   Robustly extracts JSON scores (correctness, methodology, etc.) from the judge's text response.

### `prepare_prompts.py`
Extracts high-quality prompts from distillation datasets to create standardized evaluation sets.

#### Functions
*   `prepare_prompts(...) -> pd.DataFrame`
    *   Samples prompts from source data using stratified sampling by domain to ensure balanced test coverage.

### `toolproto.py`
Specialized evaluator for tool-augmented conversations, checking for protocol compliance.

#### Classes
*   `ToolProtoEvaluator`
    *   `evaluate(conversations: Sequence[Dict]) -> EvaluationSummary`:
        *   Computes metrics such as tool call validity rate, successful execution rate, citation accuracy, and numeric sanity checks.
