# Nexa Data

> 📚 **Full Documentation**: [docs/pipelines/DATA.md](../../docs/pipelines/DATA.md)

## Overview

The `nexa_data` module acts as the central hub for data operations within the Nexa Compute infrastructure. It is responsible for:

*   **Data Ingestion:** Loading raw data from various sources.
*   **Validation:** ensuring data quality and schema compliance.
*   **Synthetic Generation:** Creating synthetic datasets for training and testing.
*   **Distillation:** managing the flow of data for knowledge distillation processes.
*   **Pipeline Orchestration:** preparing artifacts and staging data for high-performance training.

## Key Components

### `augment.py`
Provides utilities for applying augmentations to synthetic classification datasets.

#### Functions
*   `apply_augmentations(dataset: SyntheticClassificationDataset, augmentations: Dict[str, Augmentor]) -> SyntheticClassificationDataset`
    *   Applies a sequence of named augmentation functions to a dataset and logs the process.

### `data_generation_job.py`
The entry point script for running data generation jobs, specifically focusing on teacher collection for distillation.

#### Functions
*   `main()`
    *   Configures and executes the teacher collection process using environment variables and manifest configurations. It validates inputs, sets up the OpenAI client, and triggers the collection via `nexa_distill.collect_teacher`.

### `distill_materialize.py`
Handles the materialization of distilled datasets from tensor outputs to persistent formats.

#### Functions
*   `materialize_distilled_dataset(probabilities: Iterable[torch.Tensor], targets: Iterable[torch.Tensor], output_dir: Path) -> Path`
    *   Converts teacher probabilities and targets into a JSON dataset file (`distilled_dataset.json`) saved in the specified output directory.

### `prepare.py`
The core module for preparing data artifacts from configuration files. It integrates with `nexa_compute` core components to manage data lifecycles.

#### Functions
*   `prepare_from_config(config_path: Path, *, materialize_only: bool = False) -> ArtifactMeta`
    *   Initializes a `DataPipeline` from a config, materializes metadata, computes dataset summaries, and creates a versioned dataset artifact.
*   `stage_catalog_to_nvme(catalog_path: Path, destination: Path) -> ArtifactMeta`
    *   Validates a shard catalog and stages the data to a destination (typically NVMe storage) for high-speed access.
*   `build_dataloaders(config_path: Path, splits: Iterable[str])`
    *   Constructs data loaders for specified splits (e.g., train, validation) based on the provided configuration.

### `loaders/torch_loader.py`
Provides high-level wrappers for creating PyTorch-compatible data loaders.

#### Functions
*   `build_loader(config_path: Path, split: str)`
    *   Builds and returns a data loader for a specific dataset split using the `DataPipeline`.

### `filters/basic.py`
Contains fundamental filtering logic for datasets.

#### Functions
*   `filter_by_label(records: Iterable[tuple], allowed_labels: set[int])`
    *   A generator that yields records whose labels match the allowed set.
