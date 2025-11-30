# Nexa Training Pipeline

> **Scope**: Model Training, Fine-Tuning, and Optimization.
> **Modules**: `nexa_train`, `nexa_compute`

The Training Pipeline provides a unified interface for training foundation models using various backends (Axolotl, HuggingFace Trainer, FSDP). It abstracts the underlying infrastructure, allowing for seamless switching between local execution and distributed cloud training.

## Core Components

### 1. Training Backends (`nexa_train/backends/`)
*   **Axolotl**: The primary backend for efficient fine-tuning (LoRA, QLoRA). Supports complex configurations via YAML recipes.
*   **Native PyTorch**: Custom training loops for specialized architectures or research experiments.

### 2. Configuration (`nexa_train/configs/`)
Training jobs are defined by YAML configurations that specify:
*   **Model**: Base model path or HF Hub ID.
*   **Dataset**: SFT or Pretraining dataset URI.
*   **Hyperparameters**: Learning rate, batch size, epochs, scheduler.
*   **Hardware**: GPU requirements, quantization settings (4-bit/8-bit).

### 3. Orchestration (`nexa_compute/orchestration`)
Manages the lifecycle of training jobs:
*   **Launcher**: Handles environment setup and process spawning.
*   **Monitoring**: Streams logs and metrics (WandB) to the control plane.
*   **Checkpointing**: Automatically saves and uploads model checkpoints to S3/Storage.

## Usage

```bash
# Launch a training job with the baseline distillation config
bash scripts/shell/training/run_training.sh nexa_train/configs/baseline_distill.yaml true
```

