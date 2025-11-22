"""Worker registry for tracking remote GPU workers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class WorkerStatus(Enum):
    """Worker lifecycle states."""
    PROVISIONING = "provisioning"
    BOOTSTRAPPING = "bootstrapping"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class WorkerInfo:
    """Information about a registered worker."""
    worker_id: str
    ssh_host: str
    ssh_user: str = "root"
    gpu_count: int = 1
    gpu_type: Optional[str] = None
    status: WorkerStatus = WorkerStatus.PROVISIONING
    current_job_id: Optional[str] = None
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "ssh_host": self.ssh_host,
            "ssh_user": self.ssh_user,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "status": self.status.value,
            "current_job_id": self.current_job_id,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "metadata": self.metadata,
        }


class WorkerRegistry:
    """Registry for managing remote GPU workers."""
    
    def __init__(self, heartbeat_timeout: int = 300):
        """
        Initialize worker registry.
        
        Args:
            heartbeat_timeout: Seconds before considering worker dead
        """
        self.workers: Dict[str, WorkerInfo] = {}
        self.heartbeat_timeout = heartbeat_timeout
    
    def register_worker(
        self,
        worker_id: str,
        ssh_host: str,
        gpu_count: int = 1,
        gpu_type: Optional[str] = None,
        ssh_user: str = "root",
        metadata: Optional[Dict] = None,
    ) -> WorkerInfo:
        """Register a new worker."""
        worker = WorkerInfo(
            worker_id=worker_id,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            metadata=metadata or {},
        )
        self.workers[worker_id] = worker
        return worker
    
    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID."""
        return self.workers.get(worker_id)
    
    def update_status(self, worker_id: str, status: WorkerStatus) -> None:
        """Update worker status."""
        if worker_id in self.workers:
            self.workers[worker_id].status = status
    
    def heartbeat(self, worker_id: str) -> bool:
        """
        Record worker heartbeat.
        
        Returns:
            True if worker exists, False otherwise
        """
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()
            return True
        return False
    
    def assign_job(self, worker_id: str, job_id: str) -> bool:
        """Assign job to worker."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            if worker.status == WorkerStatus.IDLE:
                worker.current_job_id = job_id
                worker.status = WorkerStatus.BUSY
                return True
        return False
    
    def release_job(self, worker_id: str) -> None:
        """Release job from worker."""
        if worker_id in self.workers:
            self.workers[worker_id].current_job_id = None
            self.workers[worker_id].status = WorkerStatus.IDLE
    
    def get_available_worker(
        self,
        gpu_requirement: int = 1,
        gpu_type: Optional[str] = None
    ) -> Optional[WorkerInfo]:
        """
        Find an available worker matching requirements.
        
        Args:
            gpu_requirement: Minimum number of GPUs needed
            gpu_type: Optional GPU type filter
            
        Returns:
            Available worker or None
        """
        for worker in self.workers.values():
            if worker.status != WorkerStatus.IDLE:
                continue
            if worker.gpu_count < gpu_requirement:
                continue
            if gpu_type and worker.gpu_type != gpu_type:
                continue
            if not self._is_alive(worker):
                continue
            return worker
        return None
    
    def get_idle_workers(self, min_idle_time: int = 300) -> List[WorkerInfo]:
        """
        Get workers that have been idle for a while.
        
        Args:
            min_idle_time: Minimum idle time in seconds
            
        Returns:
            List of idle workers
        """
        now = time.time()
        idle_workers = []
        for worker in self.workers.values():
            if worker.status == WorkerStatus.IDLE:
                idle_duration = now - worker.last_heartbeat
                if idle_duration >= min_idle_time:
                    idle_workers.append(worker)
        return idle_workers
    
    def _is_alive(self, worker: WorkerInfo) -> bool:
        """Check if worker is alive based on heartbeat."""
        elapsed = time.time() - worker.last_heartbeat
        return elapsed < self.heartbeat_timeout
    
    def cleanup_dead_workers(self) -> List[str]:
        """
        Remove workers that haven't sent heartbeat.
        
        Returns:
            List of removed worker IDs
        """
        dead_workers = []
        for worker_id, worker in list(self.workers.items()):
            if not self._is_alive(worker):
                dead_workers.append(worker_id)
                del self.workers[worker_id]
        return dead_workers
    
    def list_workers(self, status: Optional[WorkerStatus] = None) -> List[WorkerInfo]:
        """List all workers, optionally filtered by status."""
        if status:
            return [w for w in self.workers.values() if w.status == status]
        return list(self.workers.values())
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove worker from registry."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            return True
        return False


__all__ = ["WorkerRegistry", "WorkerInfo", "WorkerStatus"]
