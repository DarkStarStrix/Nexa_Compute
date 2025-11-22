import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from nexa_compute.api.models import JobStatus, JobType, CreateJobRequest
from nexa_compute.api.database import JobDB

class JobManager:
    def __init__(self, db: Session):
        self.db = db

    def create_job(self, job_type: JobType, payload: Dict[str, Any], user_id: str = "default_user") -> JobDB:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        db_job = JobDB(
            job_id=job_id,
            job_type=job_type,
            user_id=user_id,
            payload=payload,
            status=JobStatus.PENDING
        )
        self.db.add(db_job)
        self.db.commit()
        self.db.refresh(db_job)
        return db_job

    def get_job(self, job_id: str) -> Optional[JobDB]:
        return self.db.query(JobDB).filter(JobDB.job_id == job_id).first()

    def list_jobs(self, skip: int = 0, limit: int = 100, status: Optional[JobStatus] = None) -> List[JobDB]:
        query = self.db.query(JobDB)
        if status:
            query = query.filter(JobDB.status == status)
        return query.order_by(JobDB.created_at.desc()).offset(skip).limit(limit).all()

    def update_job_status(self, job_id: str, status: JobStatus, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Optional[JobDB]:
        job = self.get_job(job_id)
        if not job:
            return None
        
        job.status = status
        if result:
            job.result = result
        if error:
            job.error = error
        
        self.db.commit()
        self.db.refresh(job)
        return job

    def assign_worker(self, job_id: str, worker_id: str) -> Optional[JobDB]:
        job = self.get_job(job_id)
        if not job:
            return None
        
        job.worker_id = worker_id
        job.status = JobStatus.ASSIGNED
        self.db.commit()
        self.db.refresh(job)
        return job
