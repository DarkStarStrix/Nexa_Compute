# NexaCompute Architecture

Complete system architecture and design principles for the NexaCompute platform.

## System Overview

NexaCompute is organized into modular layers so each concern can evolve independently. The platform enables rapid fine-tuning and distillation of high-value domain models with rigorous evaluation and automated feedback loops.

## Module Organization

Each module is a distinct submodule for the ML pipeline with clear responsibilities:

```
nexa_compute/
├── nexa_data/          # Data preparation, analysis, and feedback
│   ├── data_analysis/   # Jupyter notebooks and query utilities
│   ├── feedback/        # Feedback loop for data improvement
│   ├── filters/         # Data filtering
│   ├── loaders/         # PyTorch data loaders
│   ├── manifest/        # Dataset registry
│   └── schemas/         # Data schemas
│
├── nexa_distill/        # Knowledge distillation pipeline
│   ├── collect_teacher.py
│   ├── filter_pairs.py
│   ├── to_sft.py
│   ├── prompts/
│   └── utils/
│
├── nexa_train/          # Training and fine-tuning
│   ├── backends/        # Training backends (HF, etc.)
│   ├── configs/          # Training configurations
│   ├── models/           # Model registry
│   ├── optim/            # Optimizers and schedulers
│   └── sweeps/           # Hyperparameter sweeps
│
├── nexa_eval/           # Evaluation and benchmarking
│   ├── generate.py      # Generate predictions
│   ├── judge.py         # Score predictions
│   ├── analyze.py       # Aggregate metrics
│   ├── reports/          # Evaluation reports
│   ├── rubrics/          # Evaluation rubrics
│   └── tasks/            # Evaluation tasks
│
├── nexa_ui/             # Visualization and dashboards
│   ├── leaderboard.py   # Streamlit UI for data viewing
│   ├── dashboards/      # Dashboard components
│   └── static/          # Static assets
│
└── nexa_infra/          # Infrastructure and orchestration
    ├── provision.py     # Cluster provisioning
    ├── launch_job.py    # Job launching
    ├── slurm.py         # Slurm integration
    └── cost_tracker.py  # Cost tracking
```

## Layer Architecture

### Data Layer (`nexa_data/`)
- **Purpose:** Dataset registry and pipeline support raw/processed storage, metadata logging, and batched PyTorch loaders.
- **Features:**
  - Organized storage hierarchy (`data/raw/`, `data/processed/`)
  - Query interface for reliable data access
  - Dataset versioning and manifest tracking
  - Support for JSONL, Parquet, and compressed formats
  - Feedback loop for data improvement based on evaluation

### Distillation Layer (`nexa_distill/`)
- **Purpose:** Transform raw scientific text into high-quality training data via teacher-student distillation.
- **Features:**
  - Teacher completion collection
  - Quality filtering and inspection
  - SFT dataset packaging

### Training Layer (`nexa_train/`)
- **Purpose:** Training and distillation workflows with distributed support.
- **Features:**
  - `nexa_train.train` wraps core trainer with callbacks, DDP support, and manifest logging
  - Support for supervised fine-tuning and knowledge distillation
  - Automatic checkpointing and resume capability
  - Integration with W&B and MLflow

### Evaluation Layer (`nexa_eval/`)
- **Purpose:** Metric computation, artifact export, plotting, and rubric judging.
- **Features:**
  - `nexa_eval` packages metric computation and artifact export
  - LLM-as-judge evaluation for scientific tasks
  - Rubric-based scoring
  - Leaderboard generation

### UI Layer (`nexa_ui/`)
- **Purpose:** Visualization and data viewing via Streamlit.
- **Features:**
  - Streamlit dashboards for evaluation metrics
  - Visualization of distillation data
  - Training data statistics
  - Reads from organized `data/processed/` structure

### Infrastructure Layer (`nexa_infra/`)
- **Purpose:** Infrastructure provisioning, code syncing, and job launching.
- **Features:**
  - `orchestrate.py` provides unified CLI
  - Cluster provisioning and management
  - Distributed job coordination
  - Cost tracking

## Data Flow

### Training Pipeline
1. Load config and seed environment.
2. Build dataloaders through `DataPipeline`.
3. Instantiate model via registry.
4. Configure callbacks and train, emitting checkpoints/logs.
5. Evaluate best model to produce metrics and packaged artifacts.
6. Generate manifest with complete run metadata.

### Distillation Pipeline
1. Curate enhanced exemplars from raw data.
2. Generate teacher inputs with prompt templates.
3. Collect teacher completions via API.
4. Filter and score teacher outputs.
5. Package into SFT-ready dataset.
6. Train student model on distilled data.

### Evaluation Pipeline
1. Load model checkpoint.
2. Generate predictions on evaluation set.
3. Score with rubric (LLM-as-judge or heuristics).
4. Aggregate metrics and generate leaderboard.
5. Store evaluation artifacts in organized structure.

### Feedback Loop
1. Analyze evaluation results for weaknesses.
2. Generate feedback dataset targeting weak areas.
3. Incorporate feedback into data preparation.
4. Repeat training cycle with improved data.

## Data Organization

All data follows organized structure in `data/processed/`:

```
data/processed/
├── distillation/        # Distillation pipeline outputs
│   ├── teacher_inputs/
│   ├── teacher_outputs/
│   ├── filtered/
│   ├── sft_datasets/
│   └── manifests/
├── training/            # Training pipeline outputs
│   ├── train/
│   ├── val/
│   └── test/
├── evaluation/          # Evaluation outputs
│   ├── predictions/
│   ├── metrics/
│   ├── reports/
│   └── feedback/
└── raw_summary/         # Raw data analysis summaries
```

## Extensibility Points

### Registering New Components

**Datasets:**
```python
from nexa_compute.data import DatasetRegistry
registry = DatasetRegistry()
registry.register("my_dataset", my_dataset_builder)
```

**Models:**
```python
from nexa_train.models import DEFAULT_MODEL_REGISTRY
DEFAULT_MODEL_REGISTRY.register("my_model", my_model_builder)
```

**Metrics:**
```python
from nexa_compute.evaluation.metrics import MetricRegistry
registry = MetricRegistry()
registry.register("my_metric", my_metric_fn)
```

### Custom Callbacks

Define callbacks without modifying core loop:
- Logging providers (W&B, MLflow, TensorBoard)
- Custom checkpoint strategies
- Early stopping logic
- Metric aggregation

### CLI Extensions

Override CLI commands or scripts with project-specific automation:
- Custom training workflows
- Domain-specific evaluation
- Integration with external systems

## Design Principles

1. **Modularity:** Each layer can evolve independently.
2. **Registry Pattern:** Extensibility via registration, not inheritance.
3. **Configuration-Driven:** All behavior controlled via YAML configs.
4. **Reproducibility:** Complete lineage tracking via manifests.
5. **Observability:** Built-in logging and metrics aggregation.
6. **Storage Hierarchy:** Clear separation of ephemeral vs. durable storage.
7. **No Overlap:** Each directory is a distinct submodule with clear boundaries.

## Integration Points

- **MLflow:** Automatic parameter and metric logging.
- **W&B:** Optional experiment tracking.
- **HuggingFace:** Dataset loading and model hub integration.
- **AWS S3:** Durable storage backend.
- **Slurm:** Cluster job scheduling.
- **Docker:** Containerized execution environments.
- **Streamlit:** UI dashboards for data visualization.

## Performance Considerations

- **Data Loading:** Efficient PyTorch DataLoaders with caching.
- **Distributed Training:** DDP support with automatic worker coordination.
- **Checkpointing:** Incremental saves to minimize I/O overhead.
- **Storage:** Use ephemeral storage for fast I/O during training.
