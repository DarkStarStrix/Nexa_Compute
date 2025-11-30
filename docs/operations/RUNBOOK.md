# NexaCompute Runbook

> **Scope**: Operations, Execution Workflows, and Maintenance.

This document serves as the operational manual for running the NexaCompute platform.

## 1. Environment Setup

### 1.1 Turn-Key Bootstrap
For a fresh GPU node (e.g., Lambda, RunPod):

```bash
# 1. Clone repo
git clone https://github.com/nexa-ai/nexa-compute.git
cd nexa-compute

# 2. Configure secrets
cp env.example .env
vim .env  # Add OPENAI_API_KEY, etc.

# 3. Run bootstrap
bash nexa_infra/Boostrap.sh
```

### 1.2 Verification
Run the environment validator to ensure readiness:
```bash
./scripts/validate_environment.sh
```

## 2. Execution Workflows

### 2.1 Full Distillation Run
To generate a dataset from scratch and train a model:

```bash
# Launch the master orchestration script
./scripts/shell/orchestration/launch_pipeline.sh
```

This script launches the following stages in `tmux` sessions:
1.  **Data Generation**: `data_gen` session.
2.  **Filtering**: `filtering` session.
3.  **Packaging**: `packaging` session.
4.  **Training**: `training` session.

**Monitoring**:
```bash
tmux attach -t data_gen
# (Ctrl+B, D to detach)
```

### 2.2 Training-Only Run
If data is already prepared:

```bash
bash scripts/shell/training/run_training.sh nexa_train/configs/baseline_distill.yaml true
```

## 3. Maintenance

### 3.1 Disk Cleanup
Intermediate artifacts can consume significant space.
```bash
# Clean up temporary shards older than 7 days
find data/shards -name "*.tmp" -mtime +7 -delete
```

### 3.2 Log Rotation
Logs are stored in `logs/`. Rotate manually if needed:
```bash
# Compress old logs
gzip logs/*.log
```

## 4. Cost Estimates

Based on production runs (Nov 2025):

| Task | Scale | Duration | Est. Cost | Note |
| :--- | :--- | :--- | :--- | :--- |
| **QC Batch** | 1k samples | ~10 min | ~$0.75 | Validation phase |
| **Full Gen** | 100k samples | ~16-18 hrs | ~$75.00 | 256 async workers |
| **Training** | 3 epochs | ~3 hrs | ~$9.00 | 2x A100 (QLoRA) |
| **Total** | **End-to-End** | **~20 hrs** | **~$85.00** | Full pipeline |

*Estimates assume GPT-4o-mini for generation and on-demand A100 pricing.*

## 5. Troubleshooting

### Common Issues

**Issue**: `tmux: session not found`
**Fix**: The script may have failed to start the session. Check `logs/` for immediate crash reports.

**Issue**: OOM during training
**Fix**: Reduce `per_device_train_batch_size` in `nexa_train/configs/baseline_distill.yaml` and increase `gradient_accumulation_steps`.

**Issue**: Low generation throughput
**Fix**: Ensure `OPENAI_API_KEY` has sufficient tier limits. Increase concurrency in `nexa_distill/configs/distill_config.yaml`.
