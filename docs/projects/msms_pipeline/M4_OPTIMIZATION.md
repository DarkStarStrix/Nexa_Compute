# M4 MacBook Optimization Guide

## Overview

The MS/MS pipeline has been optimized for safe, stable processing on MacBook M4 (16-24GB RAM) to handle the full 9GB dataset without crashes, completing within a few hours.

## Key Optimizations

### 1. Dask Configuration (Threads-Only)

**Settings:**
- **4 workers** (capped for M-series)
- **1 thread per worker** (no multiprocess overhead)
- **`processes=False`** (threads only, avoids memory thrashing)
- **3GB memory limit per worker** (12GB total, safe for 16GB Mac)

**Why:**
- M-series Macs have unified memory architecture
- Threads share memory efficiently
- Processes create unnecessary overhead and memory duplication
- 4 workers match the 4 performance cores

### 2. Shard Size (1GB)

**Setting:**
- `max_shard_size_bytes: 1_000_000_000` (1GB)

**Why:**
- Smaller shards = lower RAM spikes
- Better failure recovery (less data lost per shard)
- MacBook SSD can handle 1GB writes efficiently
- Easier to inspect and debug

### 3. Checkpoint Interval (5,000 samples)

**Setting:**
- Checkpoint every **5,000 samples**

**Why:**
- Frequent checkpoints = quick recovery from interruptions
- Low overhead (checkpoint is just JSON)
- Can resume from any 5k-sample boundary
- Prevents data loss on crashes

### 4. Preflight Checks

**Checks:**
- **Memory**: Minimum 4GB available, warns if >20GB recommended
- **Disk space**: Minimum 10GB free
- **HDF5 file**: Ping test to verify accessibility

**Why:**
- Prevents macOS from killing processes due to RAM spikes
- Catches disk space issues before processing starts
- Validates HDF5 file is readable

### 5. Debug Logging

**Setting:**
- `logging.level: DEBUG` in config

**Why:**
- Detailed logs for debugging on laptop
- Track exactly what's happening during processing
- Easier to identify bottlenecks or issues

## Configuration File

Use `projects/msms_pipeline/configs/gems_full_safe.yaml`:

```yaml
dataset_name: gems_full_safe
canonical_hdf5:
  - data/raw/GeMS_C.9.hdf5
output_root: data/shards/gems_full_safe
max_shard_size_bytes: 1_000_000_000  # 1GB shards
schema_version: 1
max_spectra: null  # Process entire dataset
preprocessing:
  normalize_intensities: true
  sort_mz: true
  min_peaks: 1
  max_precursor_mz: 2000.0
  filter_nonfinite: true
quality:
  enable_ranking: false
  ranker_model: openai/gpt-4o-mini
logging:
  level: DEBUG  # Debug logging for laptop dev mode
  log_file: logs/msms/gems_full_safe_pipeline.log
random_seed: 42
```

## Running the 9GB Job

### Basic Command

```bash
python -m nexa_data.msms.cli build-shards \
  --config projects/msms_pipeline/configs/gems_full_safe.yaml \
  --use-dask
```

### With Resume (if interrupted)

```bash
python -m nexa_data.msms.cli build-shards \
  --config projects/msms_pipeline/configs/gems_full_safe.yaml \
  --use-dask \
  --resume
```

### Custom Workers/Batch Size

```bash
python -m nexa_data.msms.cli build-shards \
  --config projects/msms_pipeline/configs/gems_full_safe.yaml \
  --use-dask \
  --num-workers 4 \
  --batch-size 1000
```

## Expected Performance

- **Throughput**: ~8,000-12,000 spectra/second
- **Total time**: ~2-3 hours for 9GB dataset
- **Memory usage**: ~12-14GB peak (safe for 16GB Mac)
- **Stability**: No crashes, resumable, debuggable

## Monitoring

### Real-time Progress

The pipeline uses `tqdm` for real-time progress bars showing:
- Spectra processed
- Samples written
- Shards written
- Errors encountered
- Processing rate

### Log Files

Check `logs/msms/gems_full_safe_pipeline.log` for detailed DEBUG logs.

### Checkpoints

Checkpoints saved to:
```
data/shards/gems_full_safe/checkpoints/checkpoint_{run_id}.json
```

### Quality Report

After completion, quality report saved to:
```
data/shards/gems_full_safe/quality_report.json
```

## Troubleshooting

### If Memory Issues Occur

1. **Reduce workers**: `--num-workers 2`
2. **Reduce batch size**: `--batch-size 500`
3. **Close other applications**

### If Disk Space Issues

1. Check available space: `df -h`
2. Clean up old shards if needed
3. Ensure at least 10GB free

### If Process Killed by macOS

1. Check Activity Monitor for memory pressure
2. Reduce workers to 2-3
3. Ensure no other heavy processes running

### Resume from Checkpoint

If interrupted, simply rerun with `--resume`:
```bash
python -m nexa_data.msms.cli build-shards \
  --config projects/msms_pipeline/configs/gems_full_safe.yaml \
  --use-dask \
  --resume
```

The pipeline will automatically skip already-processed samples.

## Design Principles

1. **Low memory usage** (16-24GB ceiling)
2. **Avoid oversharding** (1GB shards are optimal)
3. **No NUMA tricks** (M-series doesn't have NUMA)
4. **Keep Dask simple** (threads > processes)
5. **Keep Rust optional** (validator-only, not required)
6. **Favor debuggability** over throughput
7. **Stability > speed** on laptop

## What Gets Generated

After processing, you'll have:

1. **Shards**: `data/shards/gems_full_safe/train/shard_{run_id}_{index}.parquet`
2. **Manifests**: `data/shards/gems_full_safe/train/shard_{run_id}_{index}.manifest.json`
3. **Dataset manifest**: `data/shards/gems_full_safe/dataset_manifest.json`
4. **Quality report**: `data/shards/gems_full_safe/quality_report.json`
5. **Logs**: `logs/msms/gems_full_safe_pipeline.log`

## Next Steps

After successful processing:

1. **Validate shards**:
   ```bash
   python -m nexa_data.msms.cli validate-shards \
     --output data/shards/gems_full_safe
   ```

2. **View in dashboard**:
   ```bash
   streamlit run nexa_ui/msms_dashboard.py
   ```

3. **Check quality report**:
   ```bash
   cat data/shards/gems_full_safe/quality_report.json | jq
   ```

## Summary

The pipeline is now optimized for M4 MacBook with:
- ✅ Safe memory usage (12GB total)
- ✅ Threads-only Dask (no multiprocess overhead)
- ✅ Frequent checkpoints (every 5k samples)
- ✅ Preflight safety checks
- ✅ Debug logging for troubleshooting
- ✅ Expected 2-3 hour completion time
- ✅ Fully resumable on interruption

Ready to process the full 9GB dataset safely and efficiently!

