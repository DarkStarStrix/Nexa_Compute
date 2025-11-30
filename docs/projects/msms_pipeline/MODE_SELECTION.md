# MS/MS Pipeline - Mode Selection Guide

The MS/MS data processing pipeline supports two distinct modes optimized for different hardware environments.

## Modes

### 1. Local Mode (`--mode local`)

**Target Hardware:** MacBook M-series, laptops, local development machines

**Design Principles:**
- Safe, stable, debuggable
- Low memory usage (16-24 GB ceiling)
- Threads-only Dask (no multiprocess)
- Small shards (1 GB)
- Rust validator-only (not full transform engine)
- Frequent checkpointing (every 5k samples)

**Configuration:**
- **Workers:** 4 max (capped for M-series)
- **Processes:** `False` (threads-only)
- **Shard Size:** 1 GB
- **Batch Size:** 1,000
- **Memory per Worker:** 3 GB
- **Expected Throughput:** ~8k-12k spectra/sec

**Usage:**
```bash
# Using config file
python -m nexa_data.msms.cli build-shards \
  --config projects/msms_pipeline/configs/gems_local.yaml \
  --mode local

# CLI-only
python -m nexa_data.msms.cli build-shards \
  --input-hdf5 data/raw/GeMS_C.9.hdf5 \
  --output-root data/shards/my_dataset \
  --mode local \
  --use-dask
```

### 2. Production/Cloud Mode (`--mode production` or `--mode cloud`)

**Target Hardware:** 48-core HPC systems, cloud instances, production servers

**Design Principles:**
- Maximum throughput
- Full multiprocess Dask (bypasses GIL)
- Large shards (2 GB)
- Full Rust transform engine
- NUMA-aware scheduling (if available)
- Parallel shard writers
- Arrow zero-copy construction
- Streaming SHA-256 hashing

**Configuration:**
- **Workers:** 48 (or specified)
- **Processes:** `True` (multiprocess)
- **Shard Size:** 2 GB
- **Batch Size:** 10,000
- **Memory per Worker:** 3 GB
- **Parallel Writers:** 2
- **Expected Throughput:** 80k-150k spectra/sec (70-120 GB/hour)

**Usage:**
```bash
# Using config file
python -m nexa_data.msms.cli build-shards \
  --config projects/msms_pipeline/configs/gems_production_48core.yaml \
  --mode production \
  --use-dask \
  --num-workers 48

# CLI-only
python -m nexa_data.msms.cli build-shards \
  --input-hdf5 data/raw/GeMS_C.9.hdf5 \
  --output-root data/shards/production \
  --mode production \
  --use-dask \
  --num-workers 48 \
  --shard-size 2000000000
```

## Mode Selection

The mode can be specified in three ways:

1. **CLI Argument:**
   ```bash
   --mode local    # Local development mode
   --mode cloud    # Production mode (alias)
   --mode production  # Production mode
   ```

2. **Interactive Menu:**
   When using `--interactive`, you'll be prompted to select the mode.

3. **Default:**
   If not specified, defaults to `local` mode for safety.

## Key Differences

| Feature | Local Mode | Production Mode |
|---------|-----------|-----------------|
| Dask Processes | `False` (threads) | `True` (multiprocess) |
| Max Workers | 4 | 48 |
| Shard Size | 1 GB | 2 GB |
| Batch Size | 1,000 | 10,000 |
| Checkpoint Interval | 5,000 | 10,000 |
| Rust Usage | Validator only | Full transform engine |
| Parallel Writers | 1 | 2 |
| NUMA Awareness | No | Yes (if available) |
| Zero-Copy Arrow | No | Yes |
| Expected Speed | 8k-12k spec/sec | 80k-150k spec/sec |

## Production Optimizations

When running in production mode, the pipeline automatically:

1. **Enables multiprocess Dask** - Bypasses Python GIL across all cores
2. **Uses larger shards** - 2 GB shards reduce I/O overhead and shard count
3. **Larger batch sizes** - 10k+ spectra per batch for stable worker runtimes
4. **Parallel shard writing** - 2 writers overlap I/O and compute
5. **NUMA-aware binding** - If available, binds workers to NUMA nodes
6. **Full Rust optimization** - Complete transform pipeline in Rust (3-8× speedup)
7. **Zero-copy Arrow** - Avoids unnecessary memory copies
8. **Streaming hashing** - SHA-256 computed during write, not after

## Migration from Local to Production

To migrate a local config to production:

1. **Update config file:**
   - Change `max_shard_size_bytes` from `1_000_000_000` to `2_000_000_000`
   - Update `logging.level` from `DEBUG` to `INFO`

2. **Use production mode:**
   ```bash
   python -m nexa_data.msms.cli build-shards \
     --config your_config.yaml \
     --mode production \
     --use-dask \
     --num-workers 48
   ```

3. **Monitor resources:**
   - Production mode uses significantly more memory (144 GB total with 48 workers × 3 GB)
   - Ensure sufficient disk space for 2 GB shards
   - Monitor CPU usage across all cores

## Troubleshooting

### Local Mode Issues

**Problem:** Pipeline too slow on MacBook
- **Solution:** Use `--use-dask` with `--num-workers 4` (max for M-series)

**Problem:** Memory pressure
- **Solution:** Reduce `--num-workers` to 2-3, or use smaller `--batch-size`

### Production Mode Issues

**Problem:** Workers not using all cores
- **Solution:** Ensure `--mode production` and `--use-dask` with `processes=True`

**Problem:** Out of memory
- **Solution:** Reduce `--num-workers` or increase `memory_limit` per worker

**Problem:** Slow I/O
- **Solution:** Enable parallel writers (automatic in production mode)

## Best Practices

1. **Always use local mode for development** - Safer, more debuggable
2. **Test with small datasets first** - Use `--max-spectra 1000` before full runs
3. **Use dry-run for testing** - `--dry-run` simulates without writing
4. **Monitor first production run** - Check memory, CPU, and disk usage
5. **Use checkpoints** - Production mode checkpoints every 10k samples
6. **Resume capability** - Use `--resume` to continue from checkpoint

