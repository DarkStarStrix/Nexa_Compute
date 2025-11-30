# Data Organization Policy

> **Scope**: File system layout, artifact versioning, and storage conventions.

To ensure reproducibility and reliable querying across the NexaCompute platform, strict data organization policies are enforced.

## 1. Directory Structure

All data resides in the root `data/` directory, separated by processing stage.

```text
data/
├── raw/
│   └── {project_slug}/                      # Raw input data (gitignored)
│       ├── README.md                        # Raw data overview
│       ├── DATA_INDEX.md                    # Quick reference index
│       ├── *.json / *.jsonl / *.parquet     # Source corpora
│
└── processed/
    └── {project_slug}/                      # Organized processed outputs
        ├── distillation/                    # Teacher, filtered, and SFT datasets
        ├── tool_protocol/                   # Tool-protocol episodes
        ├── training/                        # Training-ready splits
        ├── evaluation/                      # Evaluation predictions/reports
        └── raw_summary/                     # Dataset statistics
```

## 2. Versioning

All processed datasets **MUST** use versioned naming to prevent overwrites and confusion.

*   **Format**: `{purpose}_{version}.{ext}`
*   **Examples**:
    *   `teacher_inputs_v1.parquet`
    *   `train_v1.parquet`
    *   `distillation_manifest_v1.json`

## 3. Query Interface

Direct file path access is discouraged in application code. Use the `DataQuery` utility for robust access.

### Usage

```python
from nexa_data.data_analysis.query_data import DataQuery

query = DataQuery()

# Load teacher inputs
teacher_df = query.get_teacher_inputs(version="v1")

# List available datasets
datasets = query.list_available_datasets()
```

## 4. Storage Policy

*   **Local Development**: `data/` is used. Large raw files should be gitignored.
*   **Production**: Data is mounted at `/mnt/nexa_durable/datasets/`.
*   **Version Control**: Only manifests and metadata (small JSON files) are committed to git. Actual data blobs are strictly excluded.

## 5. Manifests

Every processed dataset must be accompanied by a JSON manifest in `data/processed/{category}/manifests/`.

**Manifest Schema**:
*   `schema`: Description of columns/fields.
*   `version`: Semver string.
*   `row_count`: Number of samples.
*   `created_at`: ISO 8601 timestamp.
*   `source_hash`: SHA256 of the input data used to generate this dataset.
