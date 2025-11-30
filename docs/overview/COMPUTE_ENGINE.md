# Nexa Compute Engine Architecture

> **Scope**: Core runtime, DAG execution, and Artifact management.
> **Module**: `src/nexa_compute/core`

The Compute Engine is the heart of the Nexa platform (v2 architecture). It provides a unified, artifact-first runtime for executing machine learning pipelines defined in declarative YAML.

## 1. Core Principles

1.  **Artifact-First**: Every step produces a durable, atomic artifact directory.
2.  **Declarative**: Pipelines are static YAML files, not dynamic Python scripts.
3.  **Resumable**: Execution can stop and resume anywhere based on artifact existence.
4.  **Composable**: Backends (Axolotl, HF, vLLM) are swappable adapters.

## 2. Artifact Protocol

To guarantee correctness and resumability, all runners and backends adhere to the **Artifact Protocol**:

1.  **Atomic Write**: Outputs are written to a temporary directory `<path>.tmp/`.
2.  **Metadata**: A `meta.json` file is written containing:
    *   `kind`: (e.g., `checkpoint`, `eval_report`, `dataset`)
    *   `uri`: Canonical URI
    *   `hash`: Content hash
    *   `inputs`: List of input artifact URIs
    *   `labels`: Lineage tags (run_id, git_sha)
3.  **Finalize**: The directory is atomically renamed to its final path.
4.  **Seal**: An empty `COMPLETE` file is touched.
5.  **Pointer**: Pointers (e.g., `latest`) are updated only *after* the `COMPLETE` marker exists.

## 3. DAG Engine

The DAG Engine (`core/dag.py`) orchestrates execution based on YAML definitions.

### 3.1 Caching
Every step has a computed **Cache Key**:
`hash(uses + backend + scheduler + params + inputs_hash)`

*   If an artifact exists at the output path containing `COMPLETE` and the cache key matches, the step is **SKIPPED**.
*   This allows "smart resume" without re-running expensive training jobs.

### 3.2 Pipeline Specification (`pipelines/*.yaml`)

Pipelines are defined in YAML:

```yaml
pipeline:
  name: general_e2e
  steps:
    - id: train_hf
      uses: runners.train
      backend: hf
      params:
        model: distilbert-base-uncased
        dataset: glue
      out: artifacts/checkpoints/hf_baseline

    - id: eval_hf
      uses: runners.eval
      in:
        config: nexa_train/configs/baseline.yaml
        checkpoint: artifacts/checkpoints/hf_baseline
      out: artifacts/eval/hf_baseline
```

**Fields**:
*   `id`: Unique step identifier.
*   `uses`: The handler to invoke (e.g., `runners.train`).
*   `backend`: The execution backend (e.g., `axolotl`, `vllm`).
*   `in`: Input artifacts (dependencies).
*   `out`: Output artifact location.
*   `params`: Configuration passed to the backend.

## 4. Backends & Runners

The runtime decouples *what* to do (Runner) from *how* to do it (Backend).

### Runners (`runners/`)
High-level task controllers that handle artifact bookkeeping.
*   `TrainRunner`: Orchestrates training loops.
*   `EvalRunner`: Runs evaluation suites.
*   `ServeRunner`: Manages inference servers.

### Backends (`backends/`)
Adapters for specific tools.
*   **Train**:
    *   `axolotl`: Generates Axolotl config and spawns subprocess.
    *   `hf`: Runs Hugging Face Trainer.
*   **Serve**:
    *   `vllm`: Spawns vLLM API server.
    *   `hf_runtime`: Spawns FastAPI wrapper.
*   **Schedule**:
    *   `local`: Runs as a local subprocess.
    *   `slurm`: Submits sbatch jobs (roadmap).

## 5. Registry (`core/registry.py`)

A lightweight SQLite database tracks model lineage.

*   **Register**: `register(name, uri, meta)` assigns a semantic version.
*   **Resolve**: `resolve("model:latest")` returns the physical URI.
*   **Promote**: `promote("model", "v1", "prod")` updates tags after verifying artifact integrity.

