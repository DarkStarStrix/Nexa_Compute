# NexaCompute

**ML Lab in a Box** — A self-contained, production-grade machine learning research and development platform for rapid experimentation, model training, and knowledge distillation.

---

## Overview

NexaCompute is a complete machine learning platform that packages everything you need to run sophisticated ML workflows—from data preparation to model deployment—in a single, reproducible system. Designed for researchers and practitioners who need to iterate quickly on ephemeral GPU infrastructure while maintaining rigorous reproducibility and cost awareness.

**Core Philosophy:** Everything runs on disposable compute, with durable results and complete lineage tracking. Each experiment is fully reproducible, cost-tracked, and automatically documented.

### What Makes It Different

- **Complete ML Pipeline:** Data preparation, training, distillation, evaluation, and feedback loops in one platform
- **Infrastructure-Agnostic:** Works seamlessly across GPU providers (Lambda Labs, CoreWeave, RunPod, AWS, etc.)
- **Reproducible by Design:** Every run generates manifests with complete provenance
- **Cost-Aware:** Built-in cost tracking and optimization
- **Production-Ready:** Battle-tested infrastructure and operational best practices

---

## Key Features

### **Knowledge Distillation Pipeline**
Transform raw data into high-quality training datasets via teacher-student distillation:
- Automated teacher completion collection
- Quality filtering and human-in-the-loop inspection
- SFT-ready dataset packaging
- Complete workflow from prompts to trained models

### **Data Management**
- Organized storage hierarchy (`data/raw/` → `data/processed/`)
- Query interface for reliable data access
- Dataset versioning and manifest tracking
- Support for JSONL, Parquet, and compressed formats
- Automated feedback loops for data improvement

### **Training & Evaluation**
- Distributed training with DDP support
- HuggingFace integration
- Automatic checkpointing and resume
- Evaluation with LLM-as-judge and rubric-based scoring
- Real-time monitoring and telemetry

### **Visualization & Dashboards**
- Streamlit-based UI for data exploration
- Evaluation leaderboards
- Training statistics visualization
- Distillation data inspection

### **Infrastructure Orchestration**
- One-command cluster provisioning
- Automated job launching and management
- Cost tracking and reporting
- Multi-provider support (Lambda, CoreWeave, AWS, etc.)

### **Lifecycle Coverage**
- **Pre-training** (roadmap) — large-scale corpus preparation and tokenizer support
- **Fine-tuning & SFT** — supervised instruction tuning with project-scoped datasets
- **RL / RLHF** (roadmap) — reward modelling and policy optimisation pipelines
- **Mid-training Telemetry** — checkpointing, logging, and interactive dashboards
- **Post-training & Serving** — evaluation, guardrails, and deployment controllers
- **Data Management** — curated, versioned datasets with manifests and provenance

---

## Architecture

NexaCompute is organized into **six distinct modules**, each serving a specific purpose in the ML pipeline:

```
nexa_compute/
├── projects/       # Project-scoped assets (configs, docs, manifests, pipelines)
├── nexa_data/       # Data preparation, analysis, and feedback
├── nexa_distill/    # Knowledge distillation pipeline
├── nexa_train/      # Model training and fine-tuning
├── nexa_eval/       # Evaluation and benchmarking
├── nexa_ui/         # Visualization and dashboards
└── nexa_infra/      # Infrastructure and orchestration
```

Each module is self-contained with clear boundaries, communicating via versioned data artifacts rather than direct imports. This design ensures maintainability, testability, and extensibility.

**See [Architecture Documentation](docs/Overview_of_Project/ARCHITECTURE.md) for complete details.**

### Project Organization

- Project assets live under `projects/{project_slug}/`
- Guardrails and conventions documented in `docs/conventions/`
- Active projects catalogued in `docs/projects/README.md`

---

## Quick Start

### Turn-Key Setup (Recommended)

**1. Configure API Keys**
```bash
cp .env.example .env
# Edit .env with your API keys (OpenAI, HuggingFace, W&B, etc.)
```

**2. Bootstrap GPU Node**
```bash
# On your GPU cluster (Prime Intellect, Lambda, etc.)
export TAILSCALE_AUTH_KEY="your-key"  # Optional
export SSH_PUBLIC_KEY="ssh-ed25519 ..."  # Your SSH key

# Run bootstrap
bash nexa_infra/Boostrap.sh
```

**3. Deploy Code**
```bash
# From local machine
rsync -avz --exclude='.git' . user@gpu-node:/workspace/nexa_compute/
scp .env user@gpu-node:/workspace/nexa_compute/.env
```

**4. Run Complete Pipeline**
```bash
# SSH to node
ssh user@gpu-node
cd /workspace/nexa_compute

# Install dependencies
pip install -r requirements.txt

# Run training
python orchestrate.py launch --config nexa_train/configs/baseline.yaml
```

**See [SETUP.md](SETUP.md) for complete turn-key setup guide.**

### Local Development

```bash
# Clone repository
git clone <repository-url>
cd Nexa_compute

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

### Basic Workflow

**1. Prepare Data**
```bash
orchestrate.py prepare_data --config nexa_train/configs/baseline.yaml
```

**2. Run Knowledge Distillation**
```bash
# Generate teacher inputs
jupyter notebook nexa_data/data_analysis/distill_data_overview.ipynb

# Collect teacher completions
python -m nexa_distill.collect_teacher \
  --src data/processed/scientific_assistant/distillation/teacher_inputs/teacher_inputs_v1.parquet \
  --teacher openrouter:gpt-4o

# Filter and package
python -m nexa_distill.filter_pairs
python -m nexa_distill.to_sft
```

**3. Train Model**
```bash
orchestrate.py launch --config nexa_train/configs/baseline.yaml
```

**4. Evaluate**
```bash
orchestrate.py evaluate --checkpoint <checkpoint_path>
```

**5. Visualize Results**
```bash
orchestrate.py leaderboard  # Launch Streamlit dashboard
```

**6. Serve Inference**
```bash
orchestrate.py inference <checkpoint_path>  # Start inference server
```

### Complete Distillation Pipeline

For a complete example, see the [Distillation Guide](docs/Overview_of_Project/DISTILLATION.md).

---

## Core Modules

### `nexa_data/` — Data Pipeline
Data preparation, analysis, and automated feedback loops.

- **Data Analysis:** Jupyter notebooks and query utilities (`nexa_data/data_analysis/`)
- **Feedback Loop:** Improve data based on evaluation weaknesses (`nexa_data/feedback/`)
- **Data Loaders:** PyTorch DataLoader integrations
- **Dataset Registry:** Versioned dataset management

### `nexa_distill/` — Knowledge Distillation
Transform raw data into high-quality training datasets.

- Teacher completion collection
- Quality filtering and inspection
- SFT dataset packaging
- Human-in-the-loop review interface

### `nexa_train/` — Model Training
Training and fine-tuning with distributed support.

- HuggingFace and custom training backends
- Distributed training (DDP)
- Automatic checkpointing
- Hyperparameter sweeps
- W&B and MLflow integration

### `nexa_eval/` — Evaluation
Comprehensive evaluation and benchmarking.

- LLM-as-judge evaluation
- Rubric-based scoring
- Metric aggregation
- Leaderboard generation

### `nexa_ui/` — Visualization
Streamlit dashboards for data and metrics.

- Evaluation leaderboards
- Distillation data visualization
- Training statistics
- Reads from organized `data/processed/` structure

### `nexa_inference/` — Model Serving
Production-ready inference server for trained models.

- FastAPI-based inference server
- REST API for model predictions
- Health checks and model info endpoints
- Docker-ready deployment

### `nexa_infra/` — Infrastructure
Cluster provisioning, job management, and orchestration.

- Multi-provider cluster provisioning
- Automated job launching
- Cost tracking
- Code synchronization
- One-command bootstrap script

---

## Data Organization

All data follows a clean, organized structure:

```
data/
├── raw/              # Raw input data (JSON, JSONL, Parquet)
└── processed/        # Organized outputs by purpose
    ├── distillation/ # Teacher inputs, outputs, filtered data, SFT datasets
    ├── training/     # Training splits and pretrain data
    ├── evaluation/   # Predictions, metrics, reports, feedback
    └── raw_summary/  # Analysis summaries
```

**Query Interface:**
```python
from nexa_data.data_analysis.query_data import DataQuery

query = DataQuery()
teacher_df = query.get_teacher_inputs(version="v1")
pretrain_df = query.get_pretrain_dataset(shard="001")
```

---

## Turn-Key Solution

NexaCompute is designed as a **complete turn-key solution**:

- **Bring Your Own Compute:** Works with any GPU provider (Prime Intellect, Lambda Labs, CoreWeave, AWS, etc.)
- **One-Command Bootstrap:** `bash nexa_infra/Boostrap.sh` sets up entire environment
- **API Key Management:** Configure once via `.env`, use everywhere
- **Reproducible Docker:** Consistent environments across all deployments
- **Complete Pipeline:** Data → Training → Evaluation → Inference
- **Production Ready:** Inference server included for model deployment

**See [SETUP.md](SETUP.md) for complete turn-key setup guide.**

## Documentation

Comprehensive documentation is available in `docs/Overview_of_Project/`:

- **[Setup Guide](SETUP.md)** — Complete turn-key setup instructions
- **[Quick Start](docs/Overview_of_Project/QUICK_START.md)** — Get started quickly
- **[Architecture](docs/Overview_of_Project/ARCHITECTURE.md)** — System design and principles
- **[Runbook](docs/Overview_of_Project/RUNBOOK.md)** — Operations guide
- **[Policy](docs/Overview_of_Project/POLICY.md)** — Storage, safety, and cost policies
- **[Distillation Guide](docs/Overview_of_Project/DISTILLATION.md)** — Complete distillation workflow
- **[Docker Guide](docker/README.md)** — Docker deployment instructions

---

## Requirements

- **Python:** 3.11+
- **PyTorch:** 2.1.0+
- **GPU:** NVIDIA GPU with CUDA support (recommended)
- **Dependencies:** See `requirements.txt`

### Optional

- **Jupyter:** For data analysis notebooks (`pip install jupyter`)
- **AWS CLI:** For S3 storage syncing
- **Docker:** For containerized deployment
- **Streamlit:** Already included in requirements for UI dashboards
- **W&B Account:** For experiment tracking (configure via API key)

---

## Usage Examples

### Complete Distillation Workflow

```bash
# 1. Generate teacher inputs from enhanced prompts
jupyter notebook nexa_data/data_analysis/distill_data_overview.ipynb

# 2. Collect teacher completions
python -m nexa_distill.collect_teacher \
  --src data/processed/scientific_assistant/distillation/teacher_inputs/teacher_inputs_v1.parquet \
  --teacher openrouter:gpt-4o \
  --max-samples 6000

# 3. Filter and package
python -m nexa_distill.filter_pairs
python -m nexa_distill.to_sft

# 4. Train student model
python -m nexa_train.distill \
  --dataset data/processed/scientific_assistant/distillation/sft_datasets/sft_scientific_v1.jsonl

# 5. Evaluate
orchestrate.py evaluate --checkpoint <checkpoint_path>

# 6. View results
orchestrate.py leaderboard
```

### Infrastructure Provisioning

```bash
# Provision cluster (Prime Intellect, Lambda, etc.)
orchestrate.py provision --bootstrap

# Sync code to cluster
orchestrate.py sync user@gpu-node:/workspace/nexa_compute

# Launch training job
orchestrate.py launch --config nexa_train/configs/baseline.yaml

# Teardown cluster
orchestrate.py teardown
```

### Model Inference

```bash
# Start inference server
orchestrate.py inference \
  --checkpoint data/processed/training/checkpoints/latest/final.pt \
  --port 8000

# Or via Docker
docker-compose -f docker/docker-compose.yaml --profile inference up

# Test inference
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your input here", "max_tokens": 512}'
```

### Data Analysis

```python
from nexa_data.data_analysis.query_data import DataQuery

# Query processed datasets
query = DataQuery()

# Load teacher inputs
teacher_df = query.get_teacher_inputs(version="v1")

# List available datasets
datasets = query.list_available_datasets()
```

---

## Project Structure

```
Nexa_compute/
├── nexa_data/          # Data pipeline
├── nexa_distill/       # Knowledge distillation
├── nexa_train/         # Model training
├── nexa_eval/          # Evaluation
├── nexa_ui/            # Visualization
├── nexa_infra/         # Infrastructure
├── data/               # Data storage (raw + processed)
├── docs/               # Documentation
├── scripts/            # Utility scripts
├── orchestrate.py      # Unified CLI
└── pyproject.toml      # Project configuration
```

---

## Contributing

NexaCompute follows a modular architecture where each module is self-contained. To extend:

1. **Register new datasets:** Add to `nexa_data/manifest/dataset_registry.yaml`
2. **Register new models:** Use `nexa_train/models/registry.py`
3. **Add evaluation metrics:** Extend `nexa_eval/judge.py`
4. **Custom training backends:** Implement in `nexa_train/backends/`

See [Architecture Documentation](docs/Overview_of_Project/ARCHITECTURE.md) for extensibility patterns.

---

## Support

For questions, issues, or contributions:
- Review [Documentation](docs/Overview_of_Project/README.md)
- Check [Runbook](docs/Overview_of_Project/RUNBOOK.md) for operational procedures
- See [Architecture](docs/Overview_of_Project/ARCHITECTURE.md) for design details

---

**NexaCompute** — ML Lab in a Box. Everything you need to run sophisticated ML workflows, from data to deployment.
