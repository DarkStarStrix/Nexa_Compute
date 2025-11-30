from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class JobStatus(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(str, Enum):
    GENERATE = "generate"
    AUDIT = "audit"
    DISTILL = "distill"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"

class BaseJob(BaseModel):
    job_id: str
    job_type: JobType
    user_id: str = "default_user"
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any]
    worker_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)

class CreateJobRequest(BaseModel):
    payload: Dict[str, Any]

class GenerateRequest(CreateJobRequest):
    job_type: JobType = JobType.GENERATE

class AuditRequest(CreateJobRequest):
    job_type: JobType = JobType.AUDIT

class DistillRequest(CreateJobRequest):
    job_type: JobType = JobType.DISTILL

class TrainRequest(CreateJobRequest):
    job_type: JobType = JobType.TRAIN

class EvaluateRequest(CreateJobRequest):
    job_type: JobType = JobType.EVALUATE

class DeployRequest(CreateJobRequest):
    job_type: JobType = JobType.DEPLOY

class JobResponse(BaseJob):
    pass

class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    BOOTSTRAPPING = "bootstrapping"

class WorkerInfo(BaseModel):
    worker_id: str
    hostname: str
    ip_address: Optional[str] = None
    status: WorkerStatus = WorkerStatus.IDLE
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    current_job_id: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)

class WorkerRegistration(BaseModel):
    worker_id: str
    hostname: str
    gpu_type: Optional[str] = None
    gpu_count: int = 1

class BillingResourceType(str, Enum):
    GPU_HOUR = "gpu_hour"
    TOKEN_INPUT = "token_input"
    TOKEN_OUTPUT = "token_output"
    STORAGE_GB_MONTH = "storage_gb_month"

class BillingRecord(BaseModel):
    record_id: str
    user_id: str
    job_id: Optional[str] = None
    resource_type: BillingResourceType
    quantity: float
    unit_price: float
    total_cost: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(from_attributes=True)

class BillingSummary(BaseModel):
    total_cost: float
    currency: str = "USD"
    period_start: datetime
    period_end: datetime
    usage_by_type: Dict[str, float]
    cost_by_type: Dict[str, float]
