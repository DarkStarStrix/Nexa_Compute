"""Job dispatcher for remote workers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from src.infra.workers.registry import WorkerRegistry, WorkerInfo
from src.infra.workers.ssh_executor import SSHExecutor

logger = logging.getLogger(__name__)


class JobDispatcher:
    """Dispatches jobs to remote GPU workers."""
    
    def __init__(self, worker_registry: WorkerRegistry, ssh_key_path: Optional[Path] = None):
        self.worker_registry = worker_registry
        self.ssh_key_path = ssh_key_path
    
    def dispatch_job(self, job: Dict, worker: WorkerInfo) -> bool:
        """
        Dispatch job to a specific worker.
        
        Args:
            job: Job dictionary with job_id, job_type, payload
            worker: Worker to dispatch to
            
        Returns:
            True if dispatch successful
        """
        try:
            logger.info(f"Dispatching job {job['job_id']} to worker {worker.worker_id}")
            
            # Create SSH executor for this worker
            executor = SSHExecutor(
                ssh_host=worker.ssh_host,
                ssh_user=worker.ssh_user,
                ssh_key_path=self.ssh_key_path
            )
            
            # Upload job spec to worker
            job_spec_path = Path(f"/tmp/job_{job['job_id']}.json")
            job_spec_path.write_text(json.dumps(job))
            
            remote_job_path = f"/workspace/tmp/jobs/{job['job_id']}.json"
            executor.execute(f"mkdir -p /workspace/tmp/jobs")
            executor.upload_file(job_spec_path, remote_job_path)
            
            # Execute job on worker
            command = (
                f"cd /workspace/nexa_compute && "
                f"python3 -m src.workers.remote_worker execute "
                f"--job-spec {remote_job_path}"
            )
            
            executor.execute_async(command)
            
            # Update worker status
            self.worker_registry.assign_job(worker.worker_id, job['job_id'])
            
            logger.info(f"Job {job['job_id']} dispatched successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to dispatch job {job['job_id']}: {e}")
            return False
    
    def get_job_logs(self, worker: WorkerInfo, job_id: str) -> Optional[str]:
        """Fetch job logs from worker."""
        try:
            executor = SSHExecutor(
                ssh_host=worker.ssh_host,
                ssh_user=worker.ssh_user,
                ssh_key_path=self.ssh_key_path
            )
            
            result = executor.execute(
                f"cat /workspace/tmp/logs/{job_id}.log",
                check=False
            )
            
            if result.returncode == 0:
                return result.stdout
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch logs for {job_id}: {e}")
            return None
    
    def download_artifact(
        self,
        worker: WorkerInfo,
        remote_path: str,
        local_path: Path
    ) -> bool:
        """Download artifact from worker."""
        try:
            executor = SSHExecutor(
                ssh_host=worker.ssh_host,
                ssh_user=worker.ssh_user,
                ssh_key_path=self.ssh_key_path
            )
            
            executor.download_file(remote_path, local_path)
            logger.info(f"Downloaded artifact from {worker.worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download artifact: {e}")
            return False


__all__ = ["JobDispatcher"]
