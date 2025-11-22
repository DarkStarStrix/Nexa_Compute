"""Job dispatcher integration for server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from src.infra.workers.dispatcher import JobDispatcher as InfraJobDispatcher
from src.infra.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class JobDispatcher:
    """Server-side job dispatcher (wraps infra dispatcher)."""
    
    def __init__(self, worker_registry: WorkerRegistry, ssh_key_path: Optional[Path] = None):
        self.dispatcher = InfraJobDispatcher(worker_registry, ssh_key_path)
        self.worker_registry = worker_registry
    
    async def dispatch_to_worker(self, job: Dict, worker_id: str) -> bool:
        """
        Dispatch job to worker.
        
        Args:
            job: Job dictionary
            worker_id: Target worker ID
            
        Returns:
            True if successful
        """
        worker = self.worker_registry.get_worker(worker_id)
        if not worker:
            logger.error(f"Worker {worker_id} not found")
            return False
        
        return self.dispatcher.dispatch_job(job, worker)
    
    async def get_job_logs(self, job_id: str) -> Optional[str]:
        """Fetch logs for a job."""
        # Find worker running this job
        for worker in self.worker_registry.list_workers():
            if worker.current_job_id == job_id:
                return self.dispatcher.get_job_logs(worker, job_id)
        return None


__all__ = ["JobDispatcher"]
