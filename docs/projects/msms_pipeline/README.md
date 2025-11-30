# MS/MS Data Pipeline

Deterministic pipeline for processing MS/MS HDF5 datasets into training-ready Arrow/Parquet shards.

## Documentation

- **[HOW_IT_WORKS.md](./HOW_IT_WORKS.md)** - Comprehensive guide to how the pipeline works, architecture, data flow, and quality guarantees
- **[QUALITY_TOLERANCES.md](./QUALITY_TOLERANCES.md)** - Detailed quality thresholds and tolerances
- **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** - Common issues and resolutions

## Quick Start

```bash
# Smoke test (1000 spectra)
python -m nexa_data.msms.cli build-shards \
  --config projects/msms_pipeline/configs/gems_test.yaml

# Full processing
python -m nexa_data.msms.cli build-shards \
  --config projects/msms_pipeline/configs/gems_full.yaml

# Validate shards
python -m nexa_data.msms.cli validate-shards \
  --output data/shards/gems_test

# Test determinism
python -m nexa_data.msms.cli test-determinism \
  --config projects/msms_pipeline/configs/gems_test.yaml

# View dashboard
streamlit run nexa_ui/msms_dashboard.py
```

## Testing Phases

1. **Smoke Test**: Process 1000 spectra, verify pipeline runs
2. **Determinism Test**: Run twice, verify identical outputs
3. **Data Integrity Test**: Process 10k spectra, full validation
4. **Full Processing**: Process entire dataset

See `QUALITY_TOLERANCES.md` for quality thresholds.

