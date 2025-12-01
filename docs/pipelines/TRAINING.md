# Nexa Training Pipeline

> **Scope**: Model Training, Fine-Tuning, and Optimization.
> **Modules**: `nexa_train`, `nexa_compute`

The Training Pipeline provides a unified, configuration-driven interface for training foundation models. It abstracts over specific backends (HuggingFace Trainer, Axolotl, PyTorch FSDP) to provide a consistent "Nexa Training Job" definition that can run locally or on distributed clusters.

## Core Components

### 1. Training Launcher (`nexa_train/train.py`)

The central entry point that parses configuration, sets up the environment, and dispatches the job.

*   **Config Loading**: Merges base YAML configurations with CLI overrides (e.g., `--nodes`, `--gpus-per-node`).
*   **Artifact Management**: Wraps the output directory (checkpoints, logs) in a `checkpoint` artifact with a SHA256 hash and lineage metadata.
*   **Pipeline Execution**: Initializes the `TrainingPipeline` class which manages the training loop, validation, and checkpointing.

### 2. Unified Configuration

Training jobs are defined by YAML recipes located in `nexa_train/configs/`. The schema supports:

*   **`model`**:
    *   `name`: Model architecture name (registry key).
    *   `d_model`, `n_layers`, `n_heads`: Architecture hyperparameters.
    *   `max_seq_len`: Context window size.
*   **`data`**:
    *   `remote_uri`: S3/GCS path or HuggingFace dataset ID.
    *   `dataset_version`: Specific version tag.
    *   `validation`: Split definitions.
*   **`training`**:
    *   `global_batch_size`: Total batch size across all devices.
    *   `microbatch_size_per_gpu`: Per-device batch size.
    *   `grad_accumulation_steps`: Computed automatically or manual.
    *   `max_steps` / `epochs`: Duration.
    *   `save_every_steps`, `eval_every_steps`: Checkpointing frequency.
*   **`optimizer`**:
    *   `lr`: Learning rate.
    *   `weight_decay`: Regularization.
    *   `scheduler`: Learning rate schedule (e.g., `cosine_warmup`).
*   **`cluster`**:
    *   `num_nodes`: For distributed training.
    *   `gpus_per_node`: Resource request.

### 3. Backends (`nexa_train/backends/`)

The system supports pluggable backends to handle the actual training loop:

*   **Hugging Face (`backends/hf.py`)**:
    *   Uses `transformers.Trainer` and `TRL` (Transformer Reinforcement Learning).
    *   Best for SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization).
    *   Supports LoRA/QLoRA via PEFT integration.
*   **Axolotl** (Integrated via `train-heavy` container):
    *   Optimized for full fine-tuning of large models (>20B).
    *   Handles complex FSDP / DeepSpeed configurations.

### 4. Knowledge Distillation (`nexa_train/distill.py`)

Specialized routines for training student models on teacher-generated data.

*   **`distill_teacher`**: Runs inference using a "Student" configuration to generate probabilities/logits that can be compared against Teacher soft-labels.
*   **Materialization**: Converts these logits into a `distilled_dataset.json` for KL-Divergence loss training.

### 5. Checkpointing & Artifacts

NexaCompute treats checkpoints as immutable artifacts:
1.  **Creation**: Checkpoints are saved to `artifacts/checkpoints/<run_id>`.
2.  **Manifest**: A `manifest.json` is written containing the config used, metrics (loss, perplexity), and file hashes.
3.  **Lineage**: The artifact metadata links back to the specific dataset version and codebase revision used.

## Usage Examples

**1. Basic SFT (Local)**
```bash
# Train a Falcon-3 model using QLoRA
python nexa_train/train.py --config-mode v1 \
  --run-name falcon3-sft-v1 \
  --model-size small \
  --dataset data/processed/training/sft_dataset.jsonl
```

**2. Distributed Training (Slurm)**
```bash
# Launch a 4-node job
python nexa_train/train.py --config-mode v3 \
  --nodes 4 \
  --gpus-per-node 8 \
  --global-batch 1024
```

**3. Debug/Dry Run**
```bash
# Validate config and data loading without training
python nexa_train/train.py --config-mode v1 --dry-run true
```
