# Nexa Compute & Nexa Forge

**AI Infrastructure Platform with Managed API Service**

[![Tests](https://img.shields.io/badge/tests-74%20passing-brightgreen)](tests/)
[![Linting](https://img.shields.io/badge/linting-ruff%20%2B%20mypy-blue)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-lightgrey)](pyproject.toml)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dependencies](https://img.shields.io/badge/dependencies-uv-purple)](https://github.com/astral-sh/uv)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A complete AI foundry platform for orchestrating data generation, model distillation, training, and evaluation on ephemeral GPU compute.

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
├── src/
│   └── nexa_compute/
│       ├── api/         # FastAPI backend
│       └── cli/         # CLI Entrypoint
├── docs/
│   ├── compute_plans/   # Training Configuration Templates (V1/V2/V3)
│   ├── pipelines/       # Detailed Architecture Docs
│   └── projects/        # Active Research Projects
├── sdk/                 # Python Client SDK
└── pyproject.toml       # Dependencies & Config
```

---

## Core Features

### Compute Engine
- **Unified Training CLI**: `nexa_train/train.py` supports flexible overrides and configuration modes (V1 Stability, V2 Performance, V3 Full).
- **Infrastructure as Code**: Terraform modules for AWS GPU clusters.
- **Observability**: Prometheus/Grafana stack for real-time hardware monitoring.
- **Automated Provisioning**: One-command deployment to bare metal or cloud instances.

### Managed API (Nexa Forge)
- **6 Job Types**: Generate, Audit, Distill, Train, Evaluate, Deploy.
- **Worker Orchestration**: Pull-based job queue for ephemeral workers.
- **Security**: SHA256 API keys and metered billing.

**Note**: Nexa Forge is currently under active development. Features and APIs are subject to change.

---

## Documentation

For detailed instructions on how the platform works and what each component does, please refer to the documentation:

- **[Documentation Map](docs/README.md)**: Central index for all documentation.
- **[Infrastructure Guide](docs/pipelines/INFRASTRUCTURE.md)**: Docker, Provisioning, and Hardware.
- **[Training Pipeline](docs/pipelines/TRAINING.md)**: Configuration and Execution.
- **[Data Refinery](docs/pipelines/DATA.md)**: MS/MS and Synthetic Data.
- **[Compute Plans](docs/compute_plans/README.md)**: Run Configurations.

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

**Tags**: `machine-learning`, `distributed-training`, `infrastructure-as-code`, `mlops`, `knowledge-distillation`, `fastapi`, `pytorch`, `spectral-analysis`
