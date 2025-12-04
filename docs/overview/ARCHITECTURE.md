# NexaCompute Architecture

*High-level system architecture and module dependencies per Scaling Policy Section 8.*

## System Overview

NexaCompute is a modular monorepo organized as a **directed acyclic graph (DAG)**. The system follows strict module specialization with clear language boundaries and versioned contracts.

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    Core Infrastructure                      │
│  (core: logging, storage, retry, secrets, manifests)        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (depends on)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Domain Modules                           │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Data   │→ │  Models  │→ │ Training │→ │Evaluation│  │
│  │          │  │          │  │          │  │          │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (orchestrates)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                        │
│  (orchestration: pipeline wiring, workflow engine)          │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (monitors)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Cross-Cutting Systems                     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Monitoring   │  │     API      │  │    Config    │     │
│  │ (tracing,    │  │  (FastAPI,   │  │  (loaders,   │     │
│  │  metrics,    │  │   endpoints) │  │   schemas)   │     │
│  │  alerts)     │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Core Flow (DAG)

The primary data flow follows this DAG:

```
nexa_data → nexa_distill → nexa_train → nexa_eval → nexa_inference
```

### Module Responsibilities

#### `core/` - Core Infrastructure
**Responsibility**: Provides foundational primitives (logging, storage, retry, secrets, manifests, circuit breakers, timeouts) used across all NexaCompute modules.

**Dependencies**: None (foundation layer)

**Exports**: 
- `RunManifest`, `get_git_commit` - Run tracking
- `StoragePaths`, `get_storage` - Storage management
- `RetryPolicy`, `retry_call` - Resilience patterns
- `SecretManager`, `get_secret` - Secrets management
- `configure_logging`, `get_logger` - Logging infrastructure
- `CircuitBreaker`, `execution_timeout` - Execution control

#### `data/` - Data Operations
**Responsibility**: Manages dataset lifecycle including ingestion, versioning, quality filtering, statistical analysis, and integration with Rust-powered preprocessing engines.

**Dependencies**: `core/` (for logging, storage, manifests)

**Rust Integration**: 
- `nexa_data_core` - High-performance data transforms
- `nexa_data_quality` - Quality filtering and deduplication
- `nexa_stats` - Statistical operations
- `nexa_train_pack` - Sequence packing for pretraining

#### `models/` - Model Registry
**Responsibility**: Manages model definitions, registries, and reference implementations for classification tasks (MLP, ResNet, Transformer).

**Dependencies**: `core/` (for logging)

#### `training/` - Training Engine
**Responsibility**: Orchestrates model training workflows including checkpointing, distributed execution, training callbacks, loss smoothing, and deterministic seeding.

**Dependencies**: `core/`, `models/`, `data/`

**Key Components**:
- `Trainer` - Training loop implementation
- `checkpoint` - Checkpoint save/load (idempotent)
- `distributed` - Distributed training coordination
- `callbacks` - Training lifecycle hooks
- `seed` - Deterministic seeding
- `smoothing` - Loss smoothing utilities

#### `evaluation/` - Evaluation Engine
**Responsibility**: Executes model evaluation workflows including metric computation, prediction saving, and generation of evaluation reports and visualizations.

**Dependencies**: `core/`, `models/`, `data/`

#### `orchestration/` - Pipeline Orchestration
**Responsibility**: Wires together config, data, models, training, and evaluation into complete end-to-end training pipelines with manifest tracking and distributed execution support.

**Dependencies**: All domain modules (`data/`, `models/`, `training/`, `evaluation/`), `core/`

**Key Components**:
- `TrainingPipeline` - End-to-end pipeline (idempotent with checkpoints)
- `scheduler` - Workflow scheduling
- `workflow` - DAG-based workflow definitions

#### `monitoring/` - Observability
**Responsibility**: Provides observability infrastructure including distributed tracing, Prometheus metrics, GPU monitoring, and alerting systems for production operations.

**Dependencies**: `core/` (for logging, retry)

**Key Components**:
- `tracing` - OpenTelemetry distributed tracing
- `metrics` - Prometheus metrics export
- `gpu_monitor` - GPU telemetry
- `alerts` - Alert routing and delivery

#### `config/` - Configuration Management
**Responsibility**: Loads, validates, and manages training configuration files with override support and schema validation using Pydantic.

**Dependencies**: None (foundation)

#### `api/` - Nexa Forge API
**Responsibility**: Provides FastAPI-based REST API for job management, worker orchestration, artifact storage, billing, and workflow execution in the Nexa Forge managed service.

**Dependencies**: `core/`, `monitoring/`, `orchestration/`

## Language Boundaries

### Python - Orchestration Layer
- **Responsibility**: Control flow, CLI/TUI, configs & manifests, pipeline wiring, API server logic, high-level glue
- **Must NOT**: Implement heavy data transforms or CPU-bound loops
- **Delegates to**: Rust modules for compute-heavy operations

### Rust - Kernel Layer  
- **Responsibility**: Deterministic data transforms, high-volume CPU-bound operations, packing, gating, shuffling, statistical computations, scientific preprocessing
- **Location**: `rust/nexa_data_core/`, `rust/nexa_data_quality/`, `rust/nexa_stats/`, `rust/nexa_train_pack/`
- **Interface**: Clean FFI boundaries via Python wrappers in `src/nexa_compute/data/rust_*.py`

### Bash - Environment Layer
- **Responsibility**: Environment setup, tmux wrappers, trivial glue commands
- **Location**: `nexa_infra/scripts/`
- **Must NOT**: Contain core logic or branching/computation

## Versioned Contracts

The following interfaces are versioned and stable:

- **Run Manifests** (`core/manifests.py::RunManifest`) - V4 schema
- **Dataset Manifests** (`data/versioning.py::DatasetVersion`) - Content-addressable storage
- **Config Schemas** (`config/schema.py::TrainingConfig`) - Pydantic models
- **Rust ABI Boundaries** - FFI interfaces in `data/rust_*.py`
- **Storage Paths** (`core/storage.py::StoragePaths`) - Storage policy

## Idempotency Guarantees

### Idempotent Operations
- **Pipeline execution** (`orchestration/pipeline.py::TrainingPipeline.run`) - Idempotent when resuming from checkpoint
- **Metadata materialization** (`data/pipeline.py::DataPipeline.materialize_metadata`) - Always produces same output
- **Checkpoint saving** (`training/checkpoint.py::save_checkpoint`) - Overwrites safely
- **Manifest operations** (`core/manifests.py::RunManifest.save`) - Safe to call multiple times

### Non-Idempotent Operations
- **Training without checkpoint** - Creates new run each time
- **Dataset generation** - May produce different samples

## Dependency Rules

1. **Core modules** (`core/`, `config/`) have no dependencies on domain modules
2. **Domain modules** (`data/`, `models/`, `training/`, `evaluation/`) depend only on `core/`
3. **Orchestration** depends on all domain modules
4. **Cross-cutting** (`monitoring/`, `api/`) depend downward only
5. **No cycles** - Dependency graph is acyclic

## Mental Model

To understand NexaCompute:

1. **Start with `core/`** - Foundation primitives
2. **Follow data flow** - `data/` → `models/` → `training/` → `evaluation/`
3. **Understand orchestration** - `orchestration/` wires everything together
4. **Observe cross-cutting** - `monitoring/` and `api/` provide operational capabilities
5. **Check Rust boundaries** - Heavy compute lives in `rust/` with Python wrappers

If you can't draw this architecture, it's too complex (per Scaling Policy Section 8).
