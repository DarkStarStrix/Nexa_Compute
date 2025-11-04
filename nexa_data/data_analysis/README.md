# Data Analysis

This directory contains Jupyter notebooks and utilities for analyzing and exploring NexaCompute datasets.

## Notebooks

- **`distill_data_overview.ipynb`** - Comprehensive analysis of distillation datasets:
  - Raw data file inspection and statistics
  - Enhanced prompt exemplar curation
  - Teacher input generation for distillation
  - Dataset capacity and distribution analysis
  - **Outputs**: Organized processed data in `data/processed/distillation/`

## Utilities

- **`query_data.py`** - Query interface for processed datasets:
  ```python
  from nexa_data.data_analysis.query_data import DataQuery
  query = DataQuery()
  df = query.get_teacher_inputs(version="v1")
  ```

## Usage

Notebooks should be run sequentially after ensuring the virtual environment has required dependencies:
- `pandas`
- `pyarrow`
- `matplotlib`
- `seaborn`
- `datasets` (HuggingFace)
- `zstandard` (for compressed JSONL files)

## Outputs

All outputs are organized in `data/processed/`:
- **Distillation**: `data/processed/distillation/teacher_inputs/` - Teacher request data
- **Manifests**: `data/processed/distillation/manifests/` - Dataset manifests
- **Templates**: `data/system_prompt_template.txt` - System prompt template

## Data Organization

- **Raw Data**: `data/raw/` (see `data/raw/README.md` and `DATA_INDEX.md`)
- **Processed Data**: `data/processed/` (organized by purpose, see `data/processed/README.md`)
- **Query Interface**: Use `query_data.py` for reliable data access

## Notes

- All file paths are relative to the project root
- Large datasets are loaded incrementally to avoid memory issues
- Output artifacts follow the NexaCompute storage policy
- Use versioned file names (e.g., `teacher_inputs_v1.parquet`)

