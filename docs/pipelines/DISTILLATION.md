# NexaCompute Distillation Guide

> **Scope**: Knowledge Transfer, Teacher Generation, and Data Curation.
> **Modules**: `nexa_distill`

**Nexa Distill** transforms raw scientific text into high-quality, falsifiable, and reproducible hypothesis–method pairs for supervised fine-tuning (SFT). The engine modularizes teacher generation, filtering, inspection, regeneration, and packaging into final training datasets.

## Architecture

The pipeline is designed as a series of transformations on Parquet datasets:

```
Teacher Inputs -> [Collection] -> Raw Outputs -> [Filtering] -> Clean Outputs -> [Packaging] -> SFT Dataset
                                      ^               |
                                      |               v
                                [Regeneration] <--- [Inspection]
```

## Pipeline Components

### 1. Teacher Collection (`nexa_distill/collect_teacher.py`)
Generates synthetic reasoning traces using a high-capability model (e.g., GPT-4o).

*   **Prompting**: Uses specialized templates (`prompts/hypothesis.txt`, `prompts/methodology.txt`) that enforce a specific persona ("Rigorous Scientific Assistant").
*   **Batching**: Processes inputs in chunks to manage API rate limits and costs.
*   **Metadata**: Preserves domain and task type metadata for downstream analysis.

### 2. Filtering & Quality Gates (`nexa_distill/filter_pairs.py`, `sample_gate.py`)
Ensures only high-quality data reaches the training set.

*   **Heuristic Filters**:
    *   **Length**: Drops responses that are too short (<120 chars) or too long.
    *   **Keywords**: Enforces presence of action verbs ("simulate", "measure").
    *   **Formatting**: Checks for forbidden patterns (e.g., markdown errors, refusal strings like "I cannot").
*   **SampleGate (LLM Judge)**:
    *   Uses a lighter-weight judge model to score responses on Factuality and Reasoning.
    *   **Thresholds**: Configurable cutoffs (e.g., `judge_f > 80`, `judge_r > 80`).
    *   **Safety**: Flags potentially unsafe or hallucinated content.

### 3. Regeneration Loop (`nexa_distill/regenerate_bad.py`)
A "repair shop" for failed samples. Instead of discarding valuable prompts, we retry them with:
*   **Stricter System Prompts**: Emphasizing the specific failure mode (e.g., "Be more specific about experimental conditions").
*   **Higher Capability Models**: Using a stronger teacher (e.g., `o1-preview`) for difficult prompts.

### 4. SFT Packaging (`nexa_distill/to_sft.py`)
Prepares the final artifact for the training cluster.
*   **Formatting**: Converts `{prompt, response}` pairs into the standard chat format `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.
*   **Splitting**: Creates Train/Val/Test splits.
*   **Serialization**: Writes to both JSONL (for inspection) and Parquet (for high-performance loading).

## Configuration Reference (`configs/distill_config.yaml`)

The pipeline behavior is controlled by a central YAML file:

```yaml
defaults:
  prompt_column: "user_prompt"
  context_column: "context"
  task_type_column: "template_name"

collection:
  teacher_model: "gpt-4o-mini"
  batch_size: 8
  system_prompt_path: "nexa_distill/prompts/system_toolproto.txt"
  
  # Regeneration settings
  regen_teacher_model: "o3-mini"
  regen_system_prompt_path: "nexa_distill/prompts/system_regen.txt"

storage:
  raw_dataset: "data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet"
  collected_dataset: "data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet"
  filtered_dataset: "data/processed/distillation/filtered/teacher_filtered_v1.parquet"
  regen_dataset: "data/processed/distillation/filtered/teacher_regenerated_v1.parquet"
  sft_jsonl: "data/processed/training/sft_v1.jsonl"
  sft_parquet: "data/processed/training/sft_v1.parquet"
```

## Execution Guide

**1. Full Generation:**
```bash
python -m nexa_distill.collect_teacher --config configs/distill_config.yaml
```

**2. Filter:**
```bash
python -m nexa_distill.filter_pairs --config configs/filters.yaml
```

**3. Inspect (UI):**
```bash
streamlit run nexa_distill/ui_inspect.py -- --src data/processed/distillation/filtered/teacher_filtered_v1.parquet
```

**4. Regenerate:**
```bash
python -m nexa_distill.regenerate_bad \
  --annotations data/processed/distillation/labels/rejected.jsonl \
  --dst data/processed/distillation/filtered/regen.parquet
```

**5. Package:**
```bash
python -m nexa_distill.to_sft
```
