#!/usr/bin/env bash
# ============================================================
# NexaCompute GPU Node Bootstrap Script
# Turn-key solution for setting up ML lab environment
# ============================================================

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

# Tailscale configuration (set via environment or edit here)
TS_AUTH_KEY="${TAILSCALE_AUTH_KEY:-}"
TS_HOSTNAME="nexa-gpu-$(hostname | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')"

# SSH public key (for secure access)
SSH_PUB_KEY="${SSH_PUBLIC_KEY:-}"

# ============================================================
# Step 1: System Preparation
# ============================================================

echo "[1/7] Updating system and installing base packages..."
apt-get update -y && apt-get install -y \
    curl \
    git \
    tmux \
    rsync \
    awscli \
    python3-pip \
    python3-venv \
    docker.io \
    docker-compose \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Configure Docker for GPU support
if ! docker info | grep -q nvidia; then
    echo "Configuring Docker for NVIDIA GPU support..."
    systemctl restart docker || true
fi

# ============================================================
# Step 2: Tailscale Setup
# ============================================================

if [ -n "$TS_AUTH_KEY" ]; then
    echo "[2/7] Installing and connecting Tailscale..."
    if ! command -v tailscale &>/dev/null; then
        curl -fsSL https://tailscale.com/install.sh | sh
    fi
    
    tailscale up --ssh --authkey "${TS_AUTH_KEY}" --hostname "${TS_HOSTNAME}" || \
        echo ">>> If interactive login is needed, run 'tailscale up' manually."
    
    echo "Tailscale status:"
    tailscale status || true
else
    echo "[2/7] Skipping Tailscale (TAILSCALE_AUTH_KEY not set)"
fi

# ============================================================
# Step 3: Directory Setup
# ============================================================

echo "[3/7] Creating NexaCompute directory structure..."
mkdir -p /workspace/nexa_compute
mkdir -p /workspace/tmp/{dataloader_cache,checkpoints_temp,logs_temp,wandb_offline}
mkdir -p /mnt/nexa_durable/{datasets,checkpoints,evals/reports,evals/outputs,manifests,deploy,archive}
mkdir -p /workspace/shared/{common_datasets,eval_prompts,active_jobs}
chmod -R 777 /workspace /mnt/nexa_durable /workspace/shared 2>/dev/null || true

# ============================================================
# Step 4: Python Environment
# ============================================================

echo "[4/7] Setting up Python environment..."
python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
python3 -m pip install --upgrade pynvml >/dev/null 2>&1 || true

# ============================================================
# Step 5: Environment Variables
# ============================================================

echo "[5/7] Configuring environment variables..."
cat <<'EOF' >> ~/.bashrc

# ============================================================
# NexaCompute Environment
# ============================================================
export NEXA_SCRATCH=/workspace/tmp
export NEXA_DURABLE=/mnt/nexa_durable
export NEXA_SHARED=/workspace/shared
export NEXA_REPO=/workspace/nexa_compute
export PATH=$PATH:/workspace/nexa_compute

# NCCL Configuration
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

# Performance
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

# Storage
export NEXA_S3_PREFIX=${NEXA_S3_PREFIX:-s3://nexacompute/ML_Checkpoints}

# Python
export PYTHONPATH=/workspace/nexa_compute/src:/workspace/nexa_compute
EOF

# Source immediately for current session
source ~/.bashrc || true

# ============================================================
# Step 6: SSH Configuration
# ============================================================

if [ -n "$SSH_PUB_KEY" ]; then
    echo "[6/7] Configuring SSH access..."
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    echo "$SSH_PUB_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
else
    echo "[6/7] Skipping SSH key setup (SSH_PUBLIC_KEY not set)"
fi

# ============================================================
# Step 7: Persistent Session
# ============================================================

echo "[7/7] Setting up persistent tmux session..."
tmux has-session -t nexa 2>/dev/null || \
    tmux new-session -d -s nexa "echo 'NexaCompute session ready'; bash"

# ============================================================
# Completion
# ============================================================

echo ""
echo "============================================================"
echo "✅ NexaCompute Bootstrap Complete"
echo "============================================================"
echo ""
echo "Host: $(hostname)"
if [ -n "$TS_AUTH_KEY" ]; then
    echo "Tailscale IP: $(tailscale ip -4 2>/dev/null || echo 'Not connected')"
fi
echo ""
echo "Environment Variables:"
echo "  NEXA_SCRATCH=/workspace/tmp"
echo "  NEXA_DURABLE=/mnt/nexa_durable"
echo "  NEXA_SHARED=/workspace/shared"
echo ""
echo "Next Steps:"
echo "  1. Copy your .env file with API keys to /workspace/nexa_compute/.env"
echo "  2. Clone NexaCompute repo: git clone <repo> /workspace/nexa_compute"
echo "  3. Install dependencies: cd /workspace/nexa_compute && pip install -r requirements.txt"
echo "  4. Start training: python orchestrate.py launch --config nexa_train/configs/baseline.yaml"
echo ""
echo "Session Management:"
echo "  - Attach tmux: tmux attach -t nexa"
echo "  - Detach: Ctrl+b d"
echo ""
echo "============================================================"
