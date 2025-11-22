from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class BaseJob(BaseModel):
    job_id: str
    job_type: str
    status: str  # pending, running, completed, failed
    payload: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[str] = None

class AuditRequest(BaseModel):
    dataset_uri: str

class DistillRequest(BaseModel):
    dataset_id: str
    teacher: str = "openai/gpt-4o-mini"

class TrainRequest(BaseModel):
    dataset_id: str
    model: str = "Mistral-7B"
    epochs: int = 3

class EvaluateRequest(BaseModel):
    checkpoint_id: str

class DeployRequest(BaseModel):
    checkpoint_id: str

class WorkerRegistration(BaseModel):
    """Manual worker registration request."""
    ssh_host: str
    ssh_user: str = "root"
    ssh_key_path: Optional[str] = None
    gpu_count: int = 1
    gpu_type: Optional[str] = None
    provider: str = "manual"  # "prime_intellect", "manual", etc.
    metadata: dict = {}
