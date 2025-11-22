from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from nexa_compute.api.auth import get_api_key
from nexa_compute.api.database import get_db, UserDB
from nexa_compute.api.models import (
    JobResponse, JobStatus, JobType, 
    GenerateRequest, AuditRequest, DistillRequest, 
    TrainRequest, EvaluateRequest, DeployRequest
)
from nexa_compute.api.services.job_manager import JobManager

router = APIRouter()

def get_job_manager(
    db: Session = Depends(get_db),
    user: UserDB = Depends(get_api_key)
) -> JobManager:
    return JobManager(db)

@router.post("/generate", response_model=JobResponse)
def create_generate_job(request: GenerateRequest, manager: JobManager = Depends(get_job_manager)):
    return manager.create_job(JobType.GENERATE, request.payload)

@router.post("/audit", response_model=JobResponse)
def create_audit_job(request: AuditRequest, manager: JobManager = Depends(get_job_manager)):
    return manager.create_job(JobType.AUDIT, request.payload)

@router.post("/distill", response_model=JobResponse)
def create_distill_job(request: DistillRequest, manager: JobManager = Depends(get_job_manager)):
    return manager.create_job(JobType.DISTILL, request.payload)

@router.post("/train", response_model=JobResponse)
def create_train_job(request: TrainRequest, manager: JobManager = Depends(get_job_manager)):
    return manager.create_job(JobType.TRAIN, request.payload)

@router.post("/evaluate", response_model=JobResponse)
def create_evaluate_job(request: EvaluateRequest, manager: JobManager = Depends(get_job_manager)):
    return manager.create_job(JobType.EVALUATE, request.payload)

@router.post("/deploy", response_model=JobResponse)
def create_deploy_job(request: DeployRequest, manager: JobManager = Depends(get_job_manager)):
    return manager.create_job(JobType.DEPLOY, request.payload)

@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, manager: JobManager = Depends(get_job_manager)):
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/", response_model=List[JobResponse])
def list_jobs(
    skip: int = 0, 
    limit: int = 100, 
    status: Optional[JobStatus] = None, 
    manager: JobManager = Depends(get_job_manager)
):
    return manager.list_jobs(skip, limit, status)
