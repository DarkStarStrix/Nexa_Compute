# Docker Deployment

Reproducible Docker containers for NexaCompute ML workflows.

## Quick Start

### Build Image

```bash
docker build -f docker/Dockerfile -t nexa-compute:latest .
```

### Run Training

```bash
docker run --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  nexa-compute:latest \
  python orchestrate.py launch --config nexa_train/configs/baseline.yaml
```

### Run with Docker Compose

```bash
# Training
docker-compose -f docker/docker-compose.yaml --profile training up

# Inference
docker-compose -f docker/docker-compose.yaml --profile inference up

# UI
docker-compose -f docker/docker-compose.yaml --profile ui up
```

## Environment Variables

All API keys are loaded from `.env` file or environment:

- `OPENAI_API_KEY` - For teacher generation
- `OPENROUTER_API_KEY` - Alternative teacher provider
- `HF_TOKEN` - HuggingFace model access
- `WANDB_API_KEY` - Experiment tracking
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - S3 storage

## Image Details

- **Base:** PyTorch 2.1.0 with CUDA 12.1
- **CUDA Support:** CUDA 12.4 packages available
- **Python:** 3.11+
- **GPU:** NVIDIA Container Toolkit required

## Reproducibility

The Docker image ensures:
- Consistent Python and PyTorch versions
- Reproducible CUDA environment
- All dependencies pinned
- Same environment across local and cloud

## Multi-Node Setup

For distributed training across multiple nodes:

```bash
# Node 1
docker run --gpus all --network host \
  -e MASTER_ADDR=node1-ip \
  -e MASTER_PORT=29500 \
  nexa-compute:latest python orchestrate.py launch --distributed

# Node 2
docker run --gpus all --network host \
  -e MASTER_ADDR=node1-ip \
  -e MASTER_PORT=29500 \
  nexa-compute:latest python orchestrate.py launch --distributed
```

