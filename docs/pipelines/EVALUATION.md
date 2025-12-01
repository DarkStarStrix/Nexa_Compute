# Nexa Evaluation Pipeline

> **Scope**: Model Grading, Analysis, Benchmarking, and Protocol Verification.
> **Modules**: `nexa_eval`

The Evaluation Pipeline is the quality assurance layer of NexaCompute. It uses a combination of deterministic metrics and "LLM-as-a-Judge" methodologies to assess model performance across scientific accuracy, reasoning capability, and tool usage compliance.

## 1. Generation Benchmarking (`nexa_eval/generate_responses.py`)

The first step in evaluation is generating model outputs for a standardized test set.

*   **Multi-Backend Support**:
    *   **Local**: Queries a locally running Nexa Inference Server (FastAPI).
    *   **OpenRouter**: Queries external API models (GPT-4, Claude) for baseline comparison.
*   **Concurrency**: Uses `ThreadPoolExecutor` to maximize throughput, especially for API models.
*   **Configuration**: Supports adjusting temperature, `top_p`, and `max_tokens` to test stability vs. creativity.

## 2. LLM-as-a-Judge (`nexa_eval/judge_responses.py`)

We use a strong "Teacher" model (typically GPT-4o) to grade "Student" outputs.

### The Judge
*   **Rubrics**: Defined in `nexa_eval/rubrics/`.
    *   **Correctness**: Is the science factually accurate?
    *   **Methodology**: Is the proposed scientific method sound?
    *   **Specificity**: Is the answer vague or precise?
    *   **Hallucination Risk**: Does the model cite non-existent papers?
*   **Process**:
    1.  Constructs a prompt containing the Question, the Student's Answer, and the Rubric.
    2.  Judge returns a JSON object with scores (1-5) and specific comments.
    3.  Scores are parsed and stored in Parquet format.

## 3. Tool Protocol Evaluation (`nexa_eval/toolproto.py`)

Specialized evaluation for agentic models to ensure they follow the strict `nexa_tools` interaction protocol.

*   **Syntax Validation**: Checks if tool calls match the `~~~toolcall {}~~~` regex and schema.
*   **Execution Success**: Tracks the rate of successful tool executions vs. crashes.
*   **Numeric Sanity**: Uses `nexa_tools.units.UnitConverter` to verify if the model's math is correct (e.g., does the output value match the expected unit conversion?).
*   **Citation Verification**: Checks if cited DOIs were actually fetched via the `papers.fetch` tool (preventing citation hallucination).

## 4. Analysis & Reporting (`nexa_eval/analyze_results.py`)

Aggregates raw generation and judgment data into actionable insights.

*   **Data Merging**: Joins Prompts + Outputs + Judge Scores into a single "Merged Results" dataset.
*   **Metrics Calculation**:
    *   **Win Rates**: How often does Model A beat Model B?
    *   **Score Distributions**: Mean/Median scores per domain.
    *   **Truncation Rate**: Percentage of responses cut off by token limits.
*   **Visualization**: Generates plots (saved to `results/evaluation/plots/`) for the UI dashboard.

## Usage

**1. Prepare Prompts:**
```bash
python -m nexa_eval.prepare_prompts --source data/processed/distillation/teacher_inputs
```

**2. Generate Responses:**
```bash
python -m nexa_eval.generate_responses \
  --prompts data/processed/evaluation/prompts/prompts.parquet \
  --models-config configs/eval_models.json
```

**3. Run Judge:**
```bash
python -m nexa_eval.judge_responses \
  --outputs-dir data/processed/evaluation/outputs \
  --judge-model openai/gpt-4o
```

**4. Analyze:**
```bash
python -m nexa_eval.analyze_results
```
