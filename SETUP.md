# NexaCompute Turn-Key Setup Guide

Complete guide for setting up NexaCompute as a turn-key ML lab solution on your own GPU cluster.

## Overview

NexaCompute is designed to work with **your own GPU infrastructure** from any provider (Prime Intellect, Lambda Labs, CoreWeave, RunPod, AWS, etc.). The platform provides:

- **Reproducible Docker containers** for consistent environments
- **One-command bootstrap** to configure GPU nodes
- **Complete ML pipeline** from data to inference
- **API key management** for data generation and evaluation
- **Inference server** for model deployment

## Prerequisites

- GPU cluster with NVIDIA GPUs (A100, H100, RTX 5090, etc.)
- SSH access to nodes
- Docker and NVIDIA Container Toolkit installed
- API keys for:
  - OpenAI/OpenRouter (for teacher generation)
  - HuggingFace (for model hub)
  - W&B (optional, for experiment tracking)
  - AWS S3 (optional, for checkpoint syncing)

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url> Nexa_compute
cd Nexa_compute
```

### 2. Configure API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required keys:
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY` - For teacher generation
- `HF_TOKEN` - For HuggingFace model access
- `WANDB_API_KEY` - For experiment tracking (optional)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - For S3 syncing (optional)

### 3. Bootstrap GPU Node

On your GPU node (Prime Intellect, Lambda, etc.):

```bash
# Set environment variables
export TAILSCALE_AUTH_KEY="tskey-your-key"  # Optional, for Tailscale networking
export SSH_PUBLIC_KEY="ssh-ed25519 ..."     # Your SSH public key

# Download and run bootstrap
curl -fsSL https://raw.githubusercontent.com/your-repo/Nexa_compute/main/nexa_infra/Boostrap.sh | bash
```

Or manually:

```bash
# Copy bootstrap script
scp nexa_infra/Boostrap.sh user@gpu-node:/tmp/

# SSH and run
ssh user@gpu-node "bash /tmp/Boostrap.sh"
```

### 4. Deploy Code to Node

```bash
# From your local machine
rsync -avz --exclude='.git' --exclude='.venv' \
  . user@gpu-node:/workspace/nexa_compute/

# Copy environment file
scp .env user@gpu-node:/workspace/nexa_compute/.env
```

### 5. Install Dependencies

```bash
# SSH to node
ssh user@gpu-node

# Activate environment
cd /workspace/nexa_compute
source ~/.bashrc

# Install dependencies
pip install -r requirements.txt
```

### 6. Run Complete Pipeline

```bash
# 1. Prepare data
python orchestrate.py prepare_data

# 2. Generate teacher inputs (optional)
jupyter notebook nexa_data/data_analysis/distill_data_overview.ipynb

# 3. Collect teacher completions
python -m nexa_distill.collect_teacher \
  --src data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet \
  --teacher gpt-4o

# 4. Train model
python orchestrate.py launch --config nexa_train/configs/baseline.yaml

# 5. Evaluate
python orchestrate.py evaluate --checkpoint <checkpoint_path>

# 6. Serve inference
python -m nexa_inference.server \
  --checkpoint <checkpoint_path> \
  --port 8000
```

## Docker Deployment

### Building Images

```bash
# Build production image
docker build -f docker/Dockerfile -t nexa-compute:latest .

# Or use docker-compose
docker-compose -f docker/docker-compose.yaml build
```

### Running Services

```bash
# Training service
docker-compose -f docker/docker-compose.yaml --profile training up

# Inference service
docker-compose -f docker/docker-compose.yaml --profile inference up

# UI dashboard
docker-compose -f docker/docker-compose.yaml --profile ui up
```

### Single Container

```bash
# Run training
docker run --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  nexa-compute:latest \
  python orchestrate.py launch --config nexa_train/configs/baseline.yaml
```

## Prime Intellect Setup

Prime Intellect is an ideal provider for NexaCompute. Setup steps:

### 1. Provision Cluster

```bash
# Use Prime Intellect API or web interface
# Select GPU type (A100, H100, etc.)
# Configure storage volumes
```

### 2. Bootstrap Node

```bash
# SSH to Prime Intellect node
ssh user@prime-intellect-node

# Run bootstrap
export TAILSCALE_AUTH_KEY="your-key"
curl -fsSL <bootstrap-url> | bash
```

### 3. Deploy Code

```bash
# From local machine
rsync -avz --exclude='.git' . user@prime-node:/workspace/nexa_compute/
scp .env user@prime-node:/workspace/nexa_compute/.env
```

### 4. Run Experiments

```bash
# SSH to node
ssh user@prime-node

# Start training
cd /workspace/nexa_compute
python orchestrate.py launch --config nexa_train/configs/baseline.yaml
```

## API Key Configuration

All API keys are managed via `.env` file:

```bash
# Copy template
cp .env.example .env

# Required for distillation
OPENAI_API_KEY=sk-...
# OR
OPENROUTER_API_KEY=sk-or-...

# Required for model hub
HF_TOKEN=hf_...

# Optional but recommended
WANDB_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Keys are automatically loaded by:
- Docker containers (via environment variables)
- Python scripts (via `os.getenv()`)
- Bootstrap script (for persistent setup)

## Inference Server

Serve trained models for production use:

```bash
# Start inference server
python -m nexa_inference.server \
  --checkpoint /mnt/nexa_durable/checkpoints/run_20251103/final.pt \
  --host 0.0.0.0 \
  --port 8000

# Or via Docker
docker-compose -f docker/docker-compose.yaml --profile inference up
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Inference
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum entanglement",
    "max_tokens": 512,
    "temperature": 0.7
  }'

# Model info
curl http://localhost:8000/model/info
```

## Complete Workflow Example

### End-to-End Distillation Pipeline

```bash
# 1. Setup environment
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."
cp .env.example .env  # Fill in keys

# 2. Bootstrap node (on GPU cluster)
bash nexa_infra/Boostrap.sh

# 3. Generate teacher inputs
jupyter notebook nexa_data/data_analysis/distill_data_overview.ipynb

# 4. Collect teacher completions
python -m nexa_distill.collect_teacher \
  --src data/processed/distillation/teacher_inputs/teacher_inputs_v1.parquet \
  --teacher gpt-4o \
  --max-samples 6000

# 5. Filter and package
python -m nexa_distill.filter_pairs
python -m nexa_distill.to_sft

# 6. Train student model
python orchestrate.py launch \
  --config nexa_train/configs/baseline.yaml

# 7. Evaluate
python orchestrate.py evaluate \
  --checkpoint data/processed/training/checkpoints/latest/final.pt

# 8. Run feedback loop
python orchestrate.py feedback

# 9. Serve inference
python -m nexa_inference.server \
  --checkpoint data/processed/training/checkpoints/latest/final.pt

# 10. View dashboard
orchestrate.py leaderboard
```

## Verification

Check that everything is set up correctly:

```bash
# Check environment
python orchestrate.py summary

# Test API keys
python -c "import os; print('OpenAI:', 'OK' if os.getenv('OPENAI_API_KEY') else 'MISSING')"

# Test Docker
docker run --gpus all nexa-compute:latest python --version

# Test inference
curl http://localhost:8000/health
```

## Troubleshooting

### API Keys Not Working

```bash
# Verify keys are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT SET'))"

# Check .env file exists
ls -la .env

# Reload environment
source ~/.bashrc
```

### Docker GPU Issues

```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Docker Compose GPU config
docker-compose -f docker/docker-compose.yaml config | grep -i gpu
```

### Storage Issues

```bash
# Verify directories exist
ls -la /workspace/tmp /mnt/nexa_durable /workspace/shared

# Check permissions
sudo chmod -R 777 /workspace /mnt/nexa_durable
```

## Next Steps

- Read [Quick Start Guide](docs/Overview_of_Project/QUICK_START.md)
- Review [Architecture](docs/Overview_of_Project/ARCHITECTURE.md)
- See [Distillation Guide](docs/Overview_of_Project/DISTILLATION.md) for complete workflow
- Check [Runbook](docs/Overview_of_Project/RUNBOOK.md) for operational procedures

---

**NexaCompute** — Turn-key ML lab solution. Bring your own compute, we provide the platform.

