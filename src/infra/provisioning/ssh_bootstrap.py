"""SSH-based worker bootstrap for ephemeral GPU nodes."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Bootstrap script template (adapted from nexa_infra/Boostrap.sh)
BOOTSTRAP_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Updating system and installing base packages..."
apt-get update -y && apt-get install -y \\
    curl git tmux rsync python3-pip python3-venv \\
    docker.io docker-compose nvidia-container-toolkit \\
    && rm -rf /var/lib/apt/lists/*

if ! docker info | grep -q nvidia; then
    echo "Configuring Docker for NVIDIA GPU support..."
    systemctl restart docker || true
fi

echo "[2/6] Creating NexaCompute directory structure..."
mkdir -p /workspace/nexa_compute
mkdir -p /workspace/tmp/{dataloader_cache,checkpoints_temp,logs_temp}
mkdir -p /mnt/nexa_durable/{datasets,checkpoints,evals,manifests,deploy}
chmod -R 777 /workspace /mnt/nexa_durable 2>/dev/null || true

echo "[3/6] Setting up Python environment..."
python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true

echo "[4/6] Configuring environment variables..."
cat <<'EOF' >> ~/.bashrc
export NEXA_SCRATCH=/workspace/tmp
export NEXA_DURABLE=/mnt/nexa_durable
export NEXA_REPO=/workspace/nexa_compute
export PATH=$PATH:/workspace/nexa_compute
export PYTHONPATH=/workspace/nexa_compute/src:/workspace/nexa_compute
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
EOF
source ~/.bashrc || true

echo "[5/6] Cloning repository..."
if [ ! -d "/workspace/nexa_compute/.git" ]; then
    git clone {repo_url} /workspace/nexa_compute || echo "Clone failed, will rsync instead"
fi

echo "[6/6] Installing dependencies..."
cd /workspace/nexa_compute
if [ -f "requirements.in" ]; then
    pip install -r requirements.in >/dev/null 2>&1 || true
fi

echo "✅ Bootstrap complete"
"""


class SSHBootstrap:
    """Manages SSH-based bootstrap of remote GPU workers."""
    
    def __init__(
        self,
        ssh_host: str,
        ssh_user: str = "root",
        ssh_key_path: Optional[Path] = None,
        repo_url: str = "https://github.com/your-org/nexa_compute.git"
    ):
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
        self.repo_url = repo_url
        
    def _ssh_cmd(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute SSH command on remote host."""
        ssh_args = ["ssh"]
        if self.ssh_key_path:
            ssh_args.extend(["-i", str(self.ssh_key_path)])
        ssh_args.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{self.ssh_user}@{self.ssh_host}",
            command
        ])
        logger.info(f"SSH: {command[:100]}...")
        return subprocess.run(ssh_args, capture_output=True, text=True, check=check)
    
    def _scp_upload(self, local_path: Path, remote_path: str) -> None:
        """Upload file via SCP."""
        scp_args = ["scp"]
        if self.ssh_key_path:
            scp_args.extend(["-i", str(self.ssh_key_path)])
        scp_args.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            str(local_path),
            f"{self.ssh_user}@{self.ssh_host}:{remote_path}"
        ])
        logger.info(f"SCP: {local_path} -> {remote_path}")
        subprocess.run(scp_args, check=True)
    
    def bootstrap(self, timeout: int = 600) -> bool:
        """
        Bootstrap the remote worker.
        
        Args:
            timeout: Maximum time to wait for bootstrap (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Bootstrapping worker at {self.ssh_host}")
            
            # 1. Wait for SSH to be available
            logger.info("Waiting for SSH...")
            start = time.time()
            while time.time() - start < timeout:
                result = self._ssh_cmd("echo 'SSH ready'", check=False)
                if result.returncode == 0:
                    logger.info("SSH connection established")
                    break
                time.sleep(5)
            else:
                logger.error("SSH connection timeout")
                return False
            
            # 2. Upload and execute bootstrap script
            logger.info("Uploading bootstrap script...")
            script_content = BOOTSTRAP_SCRIPT.format(repo_url=self.repo_url)
            local_script = Path("/tmp/nexa_bootstrap.sh")
            local_script.write_text(script_content)
            
            self._scp_upload(local_script, "/tmp/nexa_bootstrap.sh")
            
            logger.info("Executing bootstrap script...")
            result = self._ssh_cmd("bash /tmp/nexa_bootstrap.sh")
            logger.info(f"Bootstrap output:\n{result.stdout}")
            
            if result.returncode != 0:
                logger.error(f"Bootstrap failed:\n{result.stderr}")
                return False
            
            # 3. Verify installation
            logger.info("Verifying installation...")
            result = self._ssh_cmd("python3 -c 'import sys; print(sys.version)'")
            logger.info(f"Python version: {result.stdout.strip()}")
            
            logger.info("✅ Bootstrap successful")
            return True
            
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            return False
    
    def sync_code(self, local_repo: Path) -> bool:
        """
        Sync local repository to remote worker using rsync.
        
        Args:
            local_repo: Path to local repository
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Syncing {local_repo} to {self.ssh_host}")
            
            rsync_args = ["rsync", "-avz", "--delete"]
            if self.ssh_key_path:
                rsync_args.extend(["-e", f"ssh -i {self.ssh_key_path} -o StrictHostKeyChecking=no"])
            else:
                rsync_args.extend(["-e", "ssh -o StrictHostKeyChecking=no"])
            
            rsync_args.extend([
                "--exclude", ".git",
                "--exclude", "*.pyc",
                "--exclude", "__pycache__",
                "--exclude", ".venv",
                "--exclude", "artifacts",
                f"{local_repo}/",
                f"{self.ssh_user}@{self.ssh_host}:/workspace/nexa_compute/"
            ])
            
            result = subprocess.run(rsync_args, capture_output=True, text=True, check=True)
            logger.info("Code sync complete")
            return True
            
        except Exception as e:
            logger.error(f"Code sync failed: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies on remote worker."""
        try:
            logger.info("Installing dependencies...")
            result = self._ssh_cmd(
                "cd /workspace/nexa_compute && pip install -r requirements.in"
            )
            logger.info("Dependencies installed")
            return True
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False
    
    def start_worker(self, vps_url: str, worker_id: str) -> bool:
        """Start the remote worker process."""
        try:
            logger.info(f"Starting remote worker {worker_id}")
            cmd = (
                f"cd /workspace/nexa_compute && "
                f"nohup python3 -m src.workers.remote_worker "
                f"--vps-url {vps_url} --worker-id {worker_id} "
                f"> /workspace/tmp/worker.log 2>&1 &"
            )
            self._ssh_cmd(cmd)
            logger.info("Worker started")
            return True
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            return False


__all__ = ["SSHBootstrap", "BOOTSTRAP_SCRIPT"]
