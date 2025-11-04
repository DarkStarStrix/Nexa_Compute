# NexaCompute Production Docker Image
# Provides reproducible environment for ML workflows

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    rsync \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt docker/cuda_12.4.txt ./

# Install Python dependencies with CUDA 12.4 support
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r docker/cuda_12.4.txt || echo "CUDA 12.4 packages optional"

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH="/workspace/src:/workspace" \
    NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    OMP_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false \
    NEXA_SCRATCH=/workspace/tmp \
    NEXA_DURABLE=/mnt/nexa_durable \
    NEXA_SHARED=/workspace/shared

# Create directories
RUN mkdir -p /workspace/tmp /mnt/nexa_durable /workspace/shared /workspace/data

# Default command (can be overridden)
CMD ["python", "orchestrate.py", "--help"]
