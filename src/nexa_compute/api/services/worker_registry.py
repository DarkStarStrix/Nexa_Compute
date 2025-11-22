from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from nexa_compute.api.models import WorkerStatus, WorkerRegistration
from nexa_compute.api.database import WorkerDB, JobDB, JobStatus

class WorkerRegistry:
    def __init__(self, db: Session):
        self.db = db

    def register_worker(self, registration: WorkerRegistration, ip_address: Optional[str] = None) -> WorkerDB:
        worker = self.get_worker(registration.worker_id)
        if not worker:
            worker = WorkerDB(
                worker_id=registration.worker_id,
                hostname=registration.hostname,
                ip_address=ip_address,
                gpu_type=registration.gpu_type,
                gpu_count=registration.gpu_count,
                status=WorkerStatus.IDLE,
                last_heartbeat=datetime.utcnow()
            )
            self.db.add(worker)
        else:
            worker.hostname = registration.hostname
            worker.ip_address = ip_address
            worker.gpu_type = registration.gpu_type
            worker.gpu_count = registration.gpu_count
            worker.last_heartbeat = datetime.utcnow()
            # Don't reset status if it's busy
            if worker.status == WorkerStatus.OFFLINE:
                worker.status = WorkerStatus.IDLE
        
        self.db.commit()
        self.db.refresh(worker)
        return worker

    def get_worker(self, worker_id: str) -> Optional[WorkerDB]:
        return self.db.query(WorkerDB).filter(WorkerDB.worker_id == worker_id).first()

    def list_workers(self) -> List[WorkerDB]:
        return self.db.query(WorkerDB).all()

    def update_heartbeat(self, worker_id: str, status: Optional[WorkerStatus] = None, current_job_id: Optional[str] = None) -> bool:
        worker = self.get_worker(worker_id)
        if not worker:
            return False
        
        worker.last_heartbeat = datetime.utcnow()
        if status:
            worker.status = status
        if current_job_id is not None: # Allow clearing with empty string if needed, but usually None means no change? No, None means no job.
            # If current_job_id is passed, update it.
            worker.current_job_id = current_job_id
            
        self.db.commit()
        return True

    def get_next_job(self, worker_id: str) -> Optional[JobDB]:
        # First check if worker is already assigned a job that is pending execution
        # (This logic might need refinement based on how assignment works)
        
        # Simple logic: Find oldest pending job
        # In a real system, we'd match capabilities (GPU type)
        
        # Transaction to ensure atomic assignment
        # SQLite doesn't support select for update well, but we'll try to be safe
        
        job = self.db.query(JobDB).filter(
            JobDB.status == JobStatus.PENDING
        ).order_by(JobDB.created_at.asc()).first()
        
        if job:
            job.status = JobStatus.ASSIGNED
            job.worker_id = worker_id
            self.db.commit()
            self.db.refresh(job)
            return job
            
        return None
