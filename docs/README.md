# NexaCompute

**ML Lab in a Box** — A self-contained, production-grade machine learning research and development platform for rapid experimentation, model training, and knowledge distillation.

---

## Documentation Map

| Section | Description |
| :--- | :--- |
| **[Overview](overview/)** | Architecture, Platform Design, and Roadmap. |
| **[Nexa Forge](Nexa_forge.md)** | API-first orchestration platform guide. |
| **[Pipelines](pipelines/)** | Detailed guides for Training, Data, Distillation, and Inference modules. |
| **[Operations](operations/)** | Setup guides, Runbooks, and Quickstarts. |
| **[Projects](projects/)** | Active project documentation. |
| **[Conventions](conventions/)** | Coding standards and data organization policies. |

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

Each module is self-contained with clear boundaries, communicating via versioned data artifacts rather than direct imports.

**See [Architecture Documentation](overview/ARCHITECTURE.md) for complete details.**

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

**See [Setup Guide](operations/SETUP.md) for complete turn-key setup instructions.**

---

## Core Modules

### [Nexa Data](pipelines/DATA.md) (`nexa_data/`)
Data preparation, analysis, and automated feedback loops. Includes the MS/MS refinery and Tool Use protocol.

### [Nexa Distill](pipelines/DISTILLATION.md) (`nexa_distill/`)
Knowledge distillation pipeline. Transform raw data into high-quality training datasets.

### [Nexa Train](pipelines/TRAINING.md) (`nexa_train/`)
Model training and fine-tuning with distributed support (Axolotl, HF Trainer).

### [Nexa Eval](pipelines/EVALUATION.md) (`nexa_eval/`)
Comprehensive evaluation and benchmarking using LLM-as-a-Judge.

### [Nexa Inference](pipelines/INFERENCE.md) (`nexa_inference/`)
Production-ready inference server (vLLM) and tool controller.

### [Nexa Infra](pipelines/INFRASTRUCTURE.md) (`nexa_infra/`)
Cluster provisioning, job management, and orchestration.

---

## Support

For questions, issues, or contributions:
- Review [Runbook](operations/RUNBOOK.md) for operational procedures
- See [Architecture](overview/ARCHITECTURE.md) for design details
- Check [Conventions](conventions/) for coding standards

---

**NexaCompute** — ML Lab in a Box. Everything you need to run sophisticated ML workflows, from data to deployment.

