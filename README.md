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
# 1. Start Backend API
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uvicorn nexa_compute.api.main:app --port 8000

# 2. Start Frontend Dashboard (new terminal)
cd frontend && npm run dev

# 3. Start Mintlify Docs (new terminal)
cd docs/mintlify && npx mintlify dev

# 4. Populate Demo Data
python scripts/create_dashboard_demo.py
```

### Access Points

- **Frontend Dashboard**: <http://localhost:3000>
- **Backend API**: <http://localhost:8000/docs>
- **Documentation**: <http://localhost:3001>

## 📁 Project Structure

```
Nexa_compute/
├── src/nexa_compute/api/          # FastAPI backend
│   ├── main.py                    # Main application
│   ├── auth.py                    # API key authentication
│   ├── database.py                # SQLAlchemy models
│   ├── endpoints/                 # API routes
│   └── services/                  # Business logic
├── frontend/                      # Next.js dashboard
│   ├── app/                       # App Router pages
│   ├── components/                # React components
│   └── lib/                       # API client
├── sdk/                           # Python SDK
│   ├── nexa_forge/               # Client library
│   ├── setup.py                  # Package config
│   └── demo.py                   # Demo script
├── docs/mintlify/                # Documentation
│   ├── mint.json                 # Mintlify config
│   ├── *.mdx                     # Doc pages
│   └── logo/                     # Branding assets
└── scripts/                      # Utility scripts
```

## 🎯 Core Features

### Backend (FastAPI)

- ✅ 6 job types (generate, audit, distill, train, evaluate, deploy)
- ✅ Worker management & orchestration
- ✅ API key authentication (SHA256 hashed)
- ✅ Metered billing tracking
- ✅ Real-time job status & logs

### Frontend (Next.js)

- ✅ Real-time dashboard with metrics
- ✅ Expandable job logs
- ✅ Worker fleet monitoring
- ✅ Billing analytics with charts
- ✅ Secure API key management
- ✅ Dark theme UI

### Python SDK

- ✅ Simple client API
- ✅ All 6 job types supported
- ✅ Environment variable config
- ✅ Comprehensive documentation

### Documentation (Mintlify)

- ✅ Getting started guides
- ✅ API reference
- ✅ Architecture diagrams
- ✅ Pricing information
- ✅ SDK examples

## 💰 Freemium Model

| Feature | Free Tier | Pro Plan |
|---------|-----------|----------|
| **GPU Hours/Month** | 10 | Unlimited |
| **Concurrent Jobs** | 2 | 50 |
| **Job Retention** | 7 days | 90 days |
| **Support** | Community | Priority |
| **SLA** | None | 99.9% |
| **Price** | $0 | $99/mo + usage |

## 📦 Python SDK Usage

```python
from nexa_forge import NexaForgeClient

# Initialize client
client = NexaForgeClient(api_key="nexa_...")

# Generate data
job1 = client.generate(domain="biology", num_samples=100)

# Train model
job2 = client.train(
    model_id="llama-3-8b",
    dataset_uri="s3://bucket/data.parquet",
    epochs=3
)

# Monitor jobs
status = client.get_job(job1['job_id'])
all_jobs = client.list_jobs(limit=10)
```

## 🔐 Security

- API keys hashed with SHA256
- One-time key display on creation
- Secure modal with warnings
- Revocation support
- Ready for rate limiting

## 📊 Tech Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Next.js 16, Tailwind CSS, Recharts
- **SDK**: Python 3.11+
- **Docs**: Mintlify
- **Deployment**: Docker Compose ready

## 🚢 Deployment

### Docker Compose

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

**Built with ❤️ using FastAPI, Next.js, and Mintlify**
