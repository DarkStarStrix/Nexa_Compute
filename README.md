# Nexa Compute & Nexa Forge

**AI Infrastructure Platform with Managed API Service**

[![Tests](https://img.shields.io/badge/tests-89%20passing-brightgreen)](tests/)
[![Linting](https://img.shields.io/badge/linting-ruff%20%2B%20mypy-blue)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-V4%20(0.4.0)-lightgrey)](pyproject.toml)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dependencies](https://img.shields.io/badge/dependencies-uv-purple)](https://github.com/astral-sh/uv)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A complete AI foundry platform for orchestrating data generation, model distillation, training, and evaluation on ephemeral GPU compute. **V4 introduces Rust-powered data pipelines, run manifests, and production-grade safety features for large-scale pretraining.**

---

## Quick Start

### 1. Setup Environment

We use [uv](https://github.com/astral-sh/uv) for dependency management. The repo is configured to work with the Homebrew install at `/opt/homebrew/bin/uv`.

```bash
# optional: create a project virtualenv
/opt/homebrew/bin/uv venv .venv
source .venv/bin/activate

# install all runtime + dev dependencies deterministically
/opt/homebrew/bin/uv pip sync requirements/requirements-dev.lock

# install git hooks
pre-commit install
```

### 2. Training Workflow

**Prerequisites:**
You must bring your own compute from your preferred vendor (AWS, GCP, Lambda, Prime Intellect, etc.).
1.  Spin up a GPU instance (Ubuntu 22.04 recommended).
2.  Copy `.env.example` to `.env` and configure your services (WandB, HuggingFace, S3).

**Deploy a Training Node:**
SSH into your node and run the turn-key deployment script. This will sync your code, install dependencies, and set up a persistent workspace.
```bash
./nexa_infra/scripts/provision/deploy.sh ubuntu@gpu-node-ip
```

**Start Training (Remote or Local):**
Once attached to the remote session (tmux), you can start training immediately:
```bash
# Run V1 Stability Plan
python nexa_train/train.py --config-mode v1 --run-name my_stability_run

# Run V2 Performance Plan (Distributed)
torchrun --nproc_per_node=8 nexa_train/train.py --config-mode v2 --dry-run true
```

### 3. Run Infrastructure

**Start Backend & Dashboard:**
```bash
# Using the orchestrator script
./nexa_infra/scripts/orchestration/start_forge.sh
```

---

## Project Structure

```
Nexa_compute/
├── nexa_data/           # Data Engineering (MS/MS, Tool Use, Distillation)
├── nexa_train/          # Training Engine (Axolotl, HF Trainer)
├── nexa_distill/        # Knowledge Distillation Pipeline
├── nexa_eval/           # Evaluation & LLM-as-a-Judge
├── nexa_inference/      # vLLM Serving & Tool Controller
├── nexa_infra/          # IaC (Terraform), Monitoring, Provisioning
├── nexa_ui/             # Dashboards (Streamlit/Next.js)
├── rust/                # Rust Extensions (V4)
│   ├── nexa_data_core/  # High-performance data preprocessing
│   ├── nexa_data_quality/ # Data quality & filtering
│   ├── nexa_train_pack/  # Sequence packing for pretraining
│   └── nexa_stats/       # Statistical operations & drift detection
├── src/
│   └── nexa_compute/
│       ├── api/         # FastAPI backend
│       ├── cli/         # CLI Entrypoint
│       ├── core/        # Core Primitives (DAG, Registry, Artifacts, Manifests)
│       ├── data/        # DataOps (Versioning, Lineage, Rust bindings)
│       ├── models/      # ModelOps (Registry, Versioning)
│       ├── monitoring/  # Observability (Alerts, Metrics, Drift)
│       └── orchestration/ # Workflow Engine (Scheduler, Templates)
├── docs/
│   ├── compute_plans/   # Training Configuration Templates (V1/V2/V3/V4)
│   ├── pipelines/       # Detailed Architecture Docs
│   ├── platform/        # Platform Guide & Best Practices
│   ├── api/             # API Reference
│   └── projects/        # Active Research Projects
├── sdk/                 # Python Client SDK
└── pyproject.toml       # Dependencies & Config
```

---

## Core Features

### V4 Highlights 🚀
- **Rust-Powered Data Pipelines**: High-performance, deterministic preprocessing with `nexa_data_core`, `nexa_data_quality`, `nexa_train_pack`, and `nexa_stats`.
- **Run Manifest System**: Complete traceability with run manifests tracking configs, datasets, hardware, and metrics.
- **Dataset Versioning**: Content-addressable storage with checksums and provenance tracking.
- **Preflight Engine**: Pre-job validation for datasets, tokenizers, GPU memory, and cost estimates.
- **Checkpoint & Resume**: Standardized checkpoint protocol with automatic resume discovery.
- **Cost Guardrails**: Built-in limits for tokens, hours, and cost with graceful shutdown.

### Compute Engine
- **Unified Training CLI**: `nexa_train/train.py` supports flexible overrides and configuration modes (V1 Stability, V2 Performance, V3 Full, V4 Production).
- **Infrastructure as Code**: Terraform modules for AWS GPU clusters.
- **Observability**: Distributed tracing (OpenTelemetry), Prometheus metrics, and real-time cost tracking.
- **Automated Provisioning**: One-command deployment to bare metal or cloud instances with Spot instance support.

### Managed API (Nexa Forge)
- **Workflows**: Declarative pipeline orchestration (DAGs) with resume capability.
- **6 Job Types**: Generate, Audit, Distill, Train, Evaluate, Deploy.
- **Worker Orchestration**: Pull-based job queue for ephemeral workers.
- **Security**: SHA256 API keys and metered billing.

### MLOps & DataOps
- **Model Registry**: Full lineage tracking from dataset to deployed model.
- **Data Versioning**: Content-addressable storage for datasets with manifest-based tracking.
- **Monitoring**: Automated drift detection and A/B testing framework with Rust-powered statistics.
- **Rust Extensions**: Zero-copy data transforms, deterministic shuffling, parallel batching, and high-volume filtering.

---

## Documentation

For detailed instructions on how the platform works and what each component does, please refer to the documentation:

- **[Documentation Map](docs/README.md)**: Central index for all documentation.
- **[Platform Guide](docs/platform/README.md)**: Overview of platform capabilities.
- **[API Reference](docs/api/README.md)**: API endpoints and usage.
- **[Infrastructure Guide](docs/pipelines/INFRASTRUCTURE.md)**: Docker, Provisioning, and Hardware.
- **[Training Pipeline](docs/pipelines/TRAINING.md)**: Configuration and Execution.
- **[Data Refinery](docs/pipelines/DATA.md)**: MS/MS and Synthetic Data.
- **[Compute Plans](docs/compute_plans/README.md)**: Run Configurations (V1/V2/V3/V4).

---

## Contributing

We welcome contributions! Please review our guidelines before submitting pull requests.

See **[docs/conventions/](docs/conventions/)** for:
- Coding Standards
- Data Organization
- Naming Conventions

### Development Commands
1.  **Linting**: `ruff check .`
2.  **Testing**: `pytest tests/`
3.  **Infrastructure**: Validate Terraform with `terraform validate`.

---

**Tags**: `machine-learning`, `distributed-training`, `infrastructure-as-code`, `mlops`, `knowledge-distillation`, `fastapi`, `pytorch`, `rust`, `spectral-analysis`, `pretraining`
