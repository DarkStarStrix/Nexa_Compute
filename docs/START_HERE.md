# 🚀 START HERE - Today's Mission

## Mission: Generate 100k Dataset + Train Model

**Date:** 2025-11-04  
**Goal:** Full dataset generation (10k-100k samples) + training attempt on 2×A100  
**Status:** ✅ All infrastructure ready, awaiting execution

---

## ✅ What's Been Done

All code and configuration has been implemented:
- ✅ Judge-F and Judge-R rubrics
- ✅ SampleGate quality filtering
- ✅ Batch generation pipeline
- ✅ Tmux launchers for all jobs
- ✅ Training configuration (QLoRA, 2×A100)
- ✅ Complete orchestration scripts
- ✅ Validation tools

---

## 🎯 What You Need To Do

### Step 1: Pre-flight Check (2 minutes)

```bash
cd /Users/allanmurimiwandia/.cursor/worktrees/Nexa_compute/ifMzH

# Validate environment
./scripts/validate_environment.sh
```

**If validation fails:** Install missing dependencies shown in the output.

### Step 2: Set API Key

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your_key_here"

# Verify
echo $OPENAI_API_KEY
```

### Step 3: QC Batch (10 minutes, $0.75)

**Purpose:** Validate pipeline with small batch before scaling.

```bash
# Start QC batch (1k samples)
./scripts/tmux_data_gen.sh 1 1000
```

**This will:**
1. Create tmux session named "data_gen"
2. Set up environment automatically
3. Show you the command ready to run
4. Press ENTER to start
5. Ctrl+B then D to detach (job keeps running)

**Monitor:**
```bash
# Attach to see progress
tmux attach-session -t data_gen

# Or check logs
tail -f logs/data_gen/run_*.log
```

**Validate QC results:**
```bash
# Check metrics
cat data/processed/distillation/generation_manifest.jsonl

# View report
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/distillation/generated_batch_0000.parquet')
print(f'Samples: {len(df)}')
print(f'Has judges: {\"judge_f_response\" in df.columns and \"judge_r_response\" in df.columns}')
"
```

**QC Pass Criteria:**
- ✅ 1000 samples generated
- ✅ Judge scores present
- ✅ No crashes or errors

### Step 4: Full Generation (16-18 hours, $50-75)

**If QC passes, scale up:**

```bash
# Generate 100k samples (10 batches × 10k)
./scripts/tmux_data_gen.sh 10 10000
```

**This runs in background.** You can:
- Detach: Ctrl+B then D
- Reattach anytime: `tmux attach-session -t data_gen`
- Check progress: `tail -f logs/data_gen/run_*.log`

**Note:** This is a long-running job. Consider running overnight.

### Step 5: Filtering (5 minutes)

**After generation completes:**

```bash
# Combine batches
python -c "
import pandas as pd
from pathlib import Path
import glob

batches = sorted(glob.glob('data/processed/distillation/generated_batch_*.parquet'))
dfs = [pd.read_parquet(f) for f in batches]
combined = pd.concat(dfs, ignore_index=True)
Path('data/processed/distillation').mkdir(parents=True, exist_ok=True)
combined.to_parquet('data/processed/distillation/generated_combined.parquet', index=False)
print(f'Combined: {len(combined):,} samples')
"

# Apply SampleGate filtering
python -m nexa_distill.sample_gate \
    --input data/processed/distillation/generated_combined.parquet \
    --output data/processed/distillation/filtered_samples.parquet \
    --rejections data/processed/distillation/rejections.parquet \
    --report data/processed/distillation/filter_report.md

# Review results
cat data/processed/distillation/filter_report.md
```

### Step 6: Prepare Training Data (2 minutes)

```bash
python -c "
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

df = pd.read_parquet('data/processed/distillation/filtered_samples.parquet')
train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)

def format_sample(row):
    return f'''### Question:
{row.get(\"prompt_text\", \"\")}

### Answer:
{row.get(\"teacher_output\", \"\")}'''

train_df['text'] = train_df.apply(format_sample, axis=1)
val_df['text'] = val_df.apply(format_sample, axis=1)

output_dir = Path('data/processed/distillation')
train_df.to_parquet(output_dir / 'sft_train.parquet', index=False)
val_df.to_parquet(output_dir / 'sft_val.parquet', index=False)

print(f'Train: {len(train_df):,} samples')
print(f'Val: {len(val_df):,} samples')
"
```

### Step 7: Launch Training (3 hours, $9)

```bash
# Start training (2×A100, QLoRA)
./scripts/tmux_training.sh
```

**Monitor:**
```bash
# Attach to training session
tmux attach-session -t training

# Or check logs
tail -f logs/training/run_*.log

# GPU monitoring
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir logs/tensorboard --port 6006
```

---

## 📋 Quick Command Reference

```bash
# Check environment
./scripts/validate_environment.sh

# QC batch
./scripts/tmux_data_gen.sh 1 1000

# Full generation
./scripts/tmux_data_gen.sh 10 10000

# List sessions
tmux ls

# Attach to session
tmux attach-session -t data_gen    # or training

# Detach from session
# Press: Ctrl+B then D

# Kill session
tmux kill-session -t data_gen      # or training

# Check logs
tail -f logs/data_gen/run_*.log
tail -f logs/training/run_*.log
```

---

## 💰 Cost Estimate

| Phase | Samples | Cost | Time |
|-------|---------|------|------|
| QC Batch | 1k | $0.75 | 10 min |
| Full Gen | 100k | $50-75 | 16-18 hrs |
| Training | - | $9 | 3 hrs |
| **Total** | **100k** | **~$60-85** | **~19-21 hrs** |

---

## 🔧 Troubleshooting

### Problem: Python not found
```bash
# Use python3 explicitly
alias python=python3
```

### Problem: Missing packages
```bash
pip install pandas torch transformers peft bitsandbytes accelerate tqdm pyyaml openai
```

### Problem: API key not set
```bash
export OPENAI_API_KEY="sk-..."
```

### Problem: Tmux session exists
```bash
tmux kill-session -t data_gen
```

### Problem: Out of memory during training
Edit `nexa_train/configs/baseline_qlora.yaml`:
```yaml
per_device_train_batch_size: 2  # Reduce from 4
```

---

## 📚 Documentation

- **This file:** Quick start guide
- `QUICK_START.md` - Command reference
- `TODAY_SUMMARY.md` - Complete implementation details
- `docs/TODAY_EXECUTION_PLAN.md` - Detailed execution guide

---

## ⏱️ Timeline

**Right Now → QC Validation:** 10 minutes  
**After QC → Full Generation:** 16-18 hours  
**After Generation → Filtering:** 5 minutes  
**After Filtering → Training:** 3 hours  

**Total:** ~19-21 hours from start to trained model

---

## ✅ Success Checklist

- [ ] Environment validated
- [ ] API key set
- [ ] QC batch completed successfully
- [ ] QC results validated
- [ ] Full generation started
- [ ] Generation completed (100k samples)
- [ ] SampleGate filtering applied
- [ ] Training data prepared
- [ ] Training started
- [ ] Training monitored and validated

---

## 🚨 Important Notes

1. **QC First:** Always run QC batch before full generation
2. **Long Running:** Full generation takes 16-18 hours - use tmux
3. **Cost Aware:** Total cost ~$60-85, mostly in generation
4. **Monitor:** Check logs periodically for errors
5. **GPU Required:** Training needs 2×A100 or similar

---

## 🎯 Your First Command

```bash
cd /Users/allanmurimiwandia/.cursor/worktrees/Nexa_compute/ifMzH
./scripts/validate_environment.sh
```

**Then if validation passes:**

```bash
export OPENAI_API_KEY="your_key_here"
./scripts/tmux_data_gen.sh 1 1000
```

---

**Need help?** Check `TODAY_SUMMARY.md` or `docs/TODAY_EXECUTION_PLAN.md`

**Ready?** Run the validation command above! 🚀

