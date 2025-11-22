"""Worker management for VPS control plane."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from src.infra.workers.registry import WorkerRegistry, WorkerStatus, WorkerInfo
from src.infra.provisioning.ssh_bootstrap import SSHBootstrap

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages worker lifecycle: provision, bootstrap, teardown."""
    
    def __init__(
        self,
        worker_registry: WorkerRegistry,
        ssh_key_path: Optional[Path] = None,
        repo_url: str = "https://github.com/your-org/nexa_compute.git"
    ):
        self.worker_registry = worker_registry
        self.ssh_key_path = ssh_key_path
        self.repo_url = repo_url
    
    def provision_worker_for_job(
        self,
        job: dict,
        gpu_count: int = 1,
        gpu_type: Optional[str] = None
    ) -> Optional[WorkerInfo]:
        """
        Provision a new worker for a job.
        
        Args:
            job: Job dictionary
            gpu_count: Number of GPUs required
            gpu_type: GPU type (e.g., "A100", "H100")
            
        Returns:
            WorkerInfo if successful, None otherwise
        """
        try:
            logger.info(f"Provisioning worker for job {job['job_id']}")
            
            # Generate worker ID
            worker_id = f"worker-{uuid.uuid4().hex[:8]}"
            
            # TODO: Call Prime Intellect API to provision GPU node
            # For now, assume we have a pre-provisioned node
            ssh_host = "PLACEHOLDER_GPU_NODE_IP"
            
            # Register worker
            worker = self.worker_registry.register_worker(
                worker_id=worker_id,
                ssh_host=ssh_host,
                gpu_count=gpu_count,
                gpu_type=gpu_type,
                metadata={"job_id": job["job_id"]}
            )
            
            # Bootstrap worker
            logger.info(f"Bootstrapping worker {worker_id}")
            worker.status = WorkerStatus.BOOTSTRAPPING
            
            bootstrap = SSHBootstrap(
                ssh_host=ssh_host,
                ssh_key_path=self.ssh_key_path,
                repo_url=self.repo_url
            )
            
            if not bootstrap.bootstrap():
                logger.error(f"Bootstrap failed for {worker_id}")
                worker.status = WorkerStatus.ERROR
                return None
            
            # Sync code
            local_repo = Path(__file__).resolve().parents[3]
            if not bootstrap.sync_code(local_repo):
                logger.error(f"Code sync failed for {worker_id}")
                worker.status = WorkerStatus.ERROR
                return None
            
            # Install dependencies
            if not bootstrap.install_dependencies():
                logger.error(f"Dependency installation failed for {worker_id}")
                worker.status = WorkerStatus.ERROR
                return None
            
            # Mark as idle
            worker.status = WorkerStatus.IDLE
            logger.info(f"Worker {worker_id} ready")
            
            return worker
            
        except Exception as e:
            logger.error(f"Worker provisioning failed: {e}")
            return None
    
    def teardown_worker(self, worker_id: str) -> bool:
        """
        Teardown a worker.
        
        Args:
            worker_id: Worker to teardown
            
        Returns:
            True if successful
        """
        try:
            worker = self.worker_registry.get_worker(worker_id)
            if not worker:
                logger.warning(f"Worker {worker_id} not found")
                return False
            
            logger.info(f"Tearing down worker {worker_id}")
            
            # TODO: Call Prime Intellect API to terminate node
            
            # Remove from registry
            self.worker_registry.remove_worker(worker_id)
            
            logger.info(f"Worker {worker_id} terminated")
            return True
            
        except Exception as e:
            logger.error(f"Worker teardown failed: {e}")
            return False
    
    def cleanup_idle_workers(self, min_idle_time: int = 600) -> int:
        """
        Teardown workers that have been idle for too long.
        
        Args:
            min_idle_time: Minimum idle time in seconds
            
        Returns:
            Number of workers torn down
        """
        idle_workers = self.worker_registry.get_idle_workers(min_idle_time)
        count = 0
        
        for worker in idle_workers:
            if self.teardown_worker(worker.worker_id):
                count += 1
        
        logger.info(f"Cleaned up {count} idle workers")
        return count


__all__ = ["WorkerManager"]
