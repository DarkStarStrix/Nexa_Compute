# Nexa Distill

> 📚 **Full Documentation**: [docs/pipelines/DISTILLATION.md](../../docs/pipelines/DISTILLATION.md)

## Overview

The `nexa_distill` module implements a complete Knowledge Distillation pipeline. It is designed to transfer knowledge from capable teacher models (e.g., GPT-4o, o1-mini) to smaller student models by generating, filtering, and refining synthetic training data. The workflow supports iterative improvement through automated filtering and manual inspection.

## Key Components

### `pipeline.py`
The central orchestrator that stitches together the distillation stages. It allows stages to be run individually or as a sequence.

#### Classes
*   `DistillationPipeline`
    *   `__init__(config_path: Path | None = None)`: Initializes the pipeline with configuration.
    *   `collect_teacher(...)`: Orchestrates the teacher collection stage.
    *   `filter_teacher(...)`: Runs heuristic filters on collected data.
    *   `regenerate(...)`: Triggers regeneration for flagged samples.
    *   `package_sft(...)`: Converts the final dataset into SFT (Supervised Fine-Tuning) format.
    *   `plan_cli_commands()`: Returns the CLI commands corresponding to the pipeline stages.

### `collect_teacher.py`
Manages interactions with teacher models to generate synthetic responses.

#### Functions
*   `run_collection(args: argparse.Namespace) -> None`
    *   Executes the collection workflow: loads data, builds prompts, queries the API (e.g., OpenAI), and saves results.
*   `build_requests(...) -> List[PromptRequest]`
    *   Converts DataFrame rows into structured prompt requests using templates.
*   `resolve_prompt(...) -> str`
    *   Formats the specific prompt for a row based on task type and templates.

### `filter_pairs.py`
Applies heuristic quality gates to filter out low-quality teacher outputs.

#### Functions
*   `run_filtering(args: argparse.Namespace) -> None`
    *   Loads the raw teacher outputs, applies configured filters, and saves the passing subset.
*   `apply_filters(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame`
    *   Runs basic filters (length, forbidden words, etc.) on every row.
*   `load_filter_config(path: Path) -> FilterConfig`
    *   Loads filter parameters from a YAML configuration file.

### `sample_gate.py`
Implements advanced quality filtering using "Judge" models (LLM-as-a-Judge).

#### Classes
*   `SampleGate`
    *   `filter_dataset(...) -> FilterStats`: Filters a dataset based on judge scores (Factuality/Reasoning) and safety flags.
    *   `filter_sample(...)`: Evaluates a single sample against thresholds.
    *   `generate_report(...)`: Creates a markdown report of the filtering statistics.

### `regenerate_bad.py`
Handles the targeted regeneration of samples that were rejected during inspection or filtering.

#### Functions
*   `run_regeneration(args: argparse.Namespace) -> None`
    *   Loads rejection annotations, subsets the original data, and re-runs teacher generation for those specific items with potentially stricter prompts.

### `to_sft.py`
Prepares the final dataset for training by formatting it into standard SFT JSONL/Parquet files.

#### Functions
*   `run_packaging(args: argparse.Namespace) -> None`
    *   Merges original and regenerated data, formats it into `{input, output}` pairs, and writes the final artifacts.
*   `build_record(...)`: Constructs a single SFT record.

### `ui_inspect.py`
A Streamlit-based user interface for manually inspecting teacher outputs and creating annotations for regeneration.

#### Functions
*   `main()`: Renders the UI, handling navigation, display of context/outputs, and saving of user annotations (Accept/Reject/Regenerate).
