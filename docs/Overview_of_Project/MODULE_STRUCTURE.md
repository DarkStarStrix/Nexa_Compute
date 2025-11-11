---
title: Module Structure
slug: overview/module-structure
description: Directory-by-directory look at NexaCompute modules and their outputs.
---

# NexaCompute Module Structure

The v2 layout introduces a clear separation between the reusable runtime that
lives under `src/nexa_compute/` and the legacy top-level helper packages (such
as `nexa_data/`, `nexa_train/`, etc.). Pipelines orchestrate functionality via
artifacts instead of ad-hoc shared state.

## Core Runtime (`src/nexa_compute/`)

### `core/`
**Purpose:** Shared infrastructure primitives that all pipelines rely on.

- `artifacts.py` – Atomic artifact protocol (`meta.json`, `COMPLETE` markers, pointer promotion)
- `dag.py` – Minimal DAG engine with caching and resume support
- `registry.py` – SQLite-backed registry (`register`, `resolve`, `promote`)
- `policies.py` – Budget and retention policy stubs

### `backends/`
**Purpose:** Thin adapters around execution technologies.

- `train/axolotl.py` – Generates Axolotl configs and launches training runs
- `train/hf.py` – Wraps `nexa_train.backends.hf` and emits checkpoint artifacts
- `serve/vllm.py` – Spawns vLLM OpenAI-compatible servers with health checks
- `serve/hf_runtime.py` – FastAPI fallback using `nexa_inference`
- `schedule/local.py` – Local subprocess launcher
- `schedule/slurm.py` – Delegates to `nexa_infra.slurm`
- `schedule/k8s.py` – Stub for Kubernetes integration
- `data/parquet.py` – Parquet reader abstraction (future use)

### `runners/`
**Purpose:** High-level orchestration that selects the appropriate backend and
takes care of artifact emission.

- `train.py` – Accepts a `TrainRunSpec`, selects backend + scheduler, and writes checkpoints
- `eval.py` – Wraps `nexa_eval.analyze.evaluate_checkpoint` and emits `eval_report` artifacts
- `serve.py` – Launches serving endpoints via the serve backends, tracks handles

### `data/`
**Purpose:** Shared data utilities used by pipelines.

- `catalog.py` – Loads shard manifests, computes content hashes, partitions shards
- `staging/nvme_cache.py` – Prefetches shards to NVMe storage using the artifact protocol
- `qa/schema.py` – Record-level schema validation helpers
- `qa/dedup.py` – MinHash deduplication stub (future implementation)
- `qa/drift.py` – KL/JS drift detection stub

### `cli/`
**Purpose:** Typer CLI entry points.

- `orchestrate.py` – Implements `pipeline run/resume/viz`, `models register/resolve/promote`, and `serve start/stop`

### `config/`, `models/`, `evaluation/`, `training/`, `utils/`
These directories continue to provide configuration loading, model registries,
metric evaluators, trainer utilities, logging helpers, etc., and are reused by
the new runners.

## Legacy Operational Packages (Top-Level)

The previous monolithic packages remain for backwards compatibility and as a
source of specialized helpers:

- `nexa_data/` – Data preparation, feedback loops, dataset registry
- `nexa_distill/` – Knowledge distillation pipeline
- `nexa_train/` – Custom training launchers, configs, and registries
- `nexa_eval/` – Evaluation workflows and report generation tools
- `nexa_ui/` – Streamlit dashboards
- `nexa_infra/` – Provisioning, launch utilities, Slurm helpers
- `nexa_inference/` – FastAPI inference runtime (-- used by `backends/serve/hf_runtime.py`)

## Projects

Project-specific assets live under `projects/{project_slug}/`:

- `configs/` – Training and evaluation configs scoped to the project
- `docs/` – Specifications, notes, and post-mortems
- `manifests/` – Machine-readable project metadata
- `pipelines/` – Pipeline definitions tailored to the project

See `docs/conventions/PROJECT_STRUCTURE.md` for templated structure and guardrails.

## Pipelines and Environment Assets

- `pipelines/` – Declarative YAML pipeline definitions such as `general_e2e.yaml`
  and `tiny_axolotl_demo.yaml`. Pipelines reference step `uses` keys that map to
  modules under `src/nexa_compute/`.
- `env/` – Environment assets (`env.example` template and `axolotl_recipe.yaml` mapping)

## Data Organization & Artifacts

Artifacts are written atomically to directories that include:

```
<artifact>/
├── meta.json
├── COMPLETE
├── manifest.json (optional)
└── payload files ...
```

The established `data/processed/` layout remains in place for durable datasets
and evaluation reports:

```
data/
├── raw/
└── processed/
    ├── distillation/
    ├── training/
    ├── evaluation/
    └── raw_summary/
```

Pipelines communicate exclusively via artifact paths rather than importing each
other’s modules. Each artifact includes hashes, provenance (`inputs`), and
labels to make caching decisions deterministic.

## Integration Points

- **CLI:** `python -m nexa_compute.cli.orchestrate pipeline run pipelines/general_e2e.yaml`
- **Registry:** `python -m nexa_compute.cli.orchestrate models register ...`
- **Serving:** `python -m nexa_compute.cli.orchestrate serve start --backend vllm ...`
- **Legacy tooling:** `nexa_data`, `nexa_train`, and `nexa_eval` can still be invoked directly; the new runners call into them under the hood.

## Benefits

1. **Runtime clarity** – All orchestration code lives under `src/nexa_compute/`
2. **Artifact-first** – Reproducibility via COMPLETE markers and metadata
3. **Composable backends** – Quick swapping between HF and Axolotl, vLLM or FastAPI
4. **Extensible DAGs** – Pipelines use declarative YAML with caching/resume support
5. **Separation of concerns** – Legacy modules provide specialized logic without coupling orchestration code

