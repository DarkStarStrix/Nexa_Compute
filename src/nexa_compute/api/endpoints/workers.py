from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from nexa_compute.api.database import get_db
from nexa_compute.api.models import WorkerInfo, WorkerRegistration, JobResponse
from nexa_compute.api.services.worker_registry import WorkerRegistry

router = APIRouter()

def get_worker_registry(db: Session = Depends(get_db)) -> WorkerRegistry:
    return WorkerRegistry(db)

class HeartbeatRequest(BaseModel):
    worker_id: str
    status: Optional[str] = None
    current_job_id: Optional[str] = None

class NextJobRequest(BaseModel):
    worker_id: str

@router.post("/register", response_model=WorkerInfo)
def register_worker(
    registration: WorkerRegistration, 
    request: Request,
    registry: WorkerRegistry = Depends(get_worker_registry)
):
    client_host = request.client.host if request.client else None
    return registry.register_worker(registration, client_host)

@router.post("/heartbeat")
def heartbeat(
    heartbeat: HeartbeatRequest,
    registry: WorkerRegistry = Depends(get_worker_registry)
):
    success = registry.update_heartbeat(
        heartbeat.worker_id, 
        heartbeat.status, 
        heartbeat.current_job_id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Worker not found")
    return {"status": "ok"}

@router.post("/next_job", response_model=Optional[JobResponse])
def get_next_job(
    request: NextJobRequest,
    registry: WorkerRegistry = Depends(get_worker_registry)
):
    job = registry.get_next_job(request.worker_id)
    if not job:
        return None # 200 OK with null body
    return job

@router.get("/", response_model=List[WorkerInfo])
def list_workers(registry: WorkerRegistry = Depends(get_worker_registry)):
    return registry.list_workers()
