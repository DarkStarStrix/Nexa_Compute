---
title: Pipeline Guide
slug: pipeline/guide
description: Step-by-step instructions for executing the full NexaCompute pipeline.
---

# NexaCompute Pipeline Execution Guide

## Overview

This guide explains how to run the complete NexaCompute pipeline for dataset generation and training using tmux sessions.

## Pipeline Stages

1. **Data Generation** (`data_gen`): Generate teacher outputs for full dataset
2. **Filtering** (`filtering`): Apply basic filters + SampleGate quality gates
3. **Packaging** (`packaging`): Convert filtered data to SFT format
4. **Training** (`training`): Train model on distilled dataset

## Quick Start

### Launch All Jobs

```bash
bash scripts/shell/orchestration/launch_pipeline.sh
```

This will create tmux sessions for each stage and start them automatically.

### Manual Execution

If you prefer to run stages manually:

#### 1. Data Generation
```bash
tmux new -s data_gen
python scripts/python/data_processing/run_full_data_gen.py
```

#### 2. Filtering
```bash
tmux new -s filtering
python scripts/python/data_processing/run_filtering.py
```

#### 3. Packaging
```bash
tmux new -s packaging
python scripts/python/deployment/run_packaging.py
```

#### 4. Training
```bash
tmux new -s training
bash scripts/shell/training/run_training.sh nexa_train/configs/baseline_distill.yaml true
```

## Tmux Session Management

### List Sessions
```bash
tmux list-sessions
```

### Attach to Session
```bash
tmux attach -t data_gen      # Data generation
tmux attach -t filtering     # Filtering
tmux attach -t packaging     # Packaging
tmux attach -t training      # Training
```

### Detach from Session
Press `Ctrl+B` then `D` (while inside tmux)

### Kill Session
```bash
tmux kill-session -t data_gen
```

## File Locations

### Inputs
- Teacher inputs: `data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet`
- System prompt: `data/system_prompt_template.txt`

### Outputs
- Teacher outputs: `data/processed/distillation/teacher_outputs/teacher_outputs_v1.parquet`
- Filtered data: `data/processed/distillation/filtered/filtered_v1.parquet`
- Rejections: `data/processed/distillation/filtered/rejections.parquet`
- SFT dataset: `data/processed/training/sft_dataset.jsonl`

### Logs
- Data generation: `logs/data_gen.log`
- Filtering: `logs/filtering.log`
- Packaging: `logs/packaging.log`
- Training: `logs/training.log`

## Configuration

### Distillation Config
- Location: `nexa_distill/configs/distill_config.yaml`
- Model: `gpt-4o-mini`
- Batch size: 8

### Filter Config
- Location: `nexa_distill/configs/filters.yaml`
- Min judge score: 0.80 (set in SampleGate)

### Training Config
- Location: `nexa_train/configs/baseline_distill.yaml`
- Distributed: 2 GPUs
- Mixed precision: Enabled

## Environment Variables

Ensure `.env` file contains:
```bash
OPENAI_API_KEY=sk-...
```

## Troubleshooting

### Session Already Exists
If a session already exists, the launch script will attach to it instead of creating a new one. Kill it first if needed:
```bash
tmux kill-session -t session_name
```

### Check Job Status
```bash
# Inside tmux session
tail -f logs/data_gen.log
```

### Monitor GPU Usage (for training)
```bash
watch -n 1 nvidia-smi
```

