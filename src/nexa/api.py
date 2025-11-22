import sys
import threading
import queue
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.server.models import (
    AuditRequest, DistillRequest, TrainRequest, 
    EvaluateRequest, DeployRequest, BaseJob, WorkerRegistration
)
from src.server.config import Config
from src.workers.worker import process_job
from src.infra.workers.registry import WorkerRegistry, WorkerStatus
from src.server.worker_manager import WorkerManager
from src.server.job_dispatcher import JobDispatcher
from src.infra.storage.backends import get_storage_backend

app = FastAPI(title="Nexa API", version="0.2.0")

# Load configuration
config = Config()
missing = config.validate()
if missing:
    print(f"⚠️  Missing configuration: {', '.join(missing)}")
    print("   Some features may not work without proper configuration")

# Initialize storage backend
storage_backend_type = config.get_storage_backend()
if storage_backend_type == "do_spaces":
    storage = get_storage_backend(
        "do_spaces",
        bucket=config.DO_SPACES_BUCKET,
        access_key=config.DO_SPACES_KEY,
        secret_key=config.DO_SPACES_SECRET,
        region=config.DO_SPACES_REGION,
        endpoint_url=config.DO_SPACES_ENDPOINT
    )
    print(f"✅ Using DigitalOcean Spaces: {config.DO_SPACES_BUCKET}")
elif storage_backend_type == "s3":
    storage = get_storage_backend(
        "s3",
        bucket=config.S3_BUCKET,
        access_key=config.AWS_ACCESS_KEY_ID,
        secret_key=config.AWS_SECRET_ACCESS_KEY,
        region=config.S3_REGION
    )
    print(f"✅ Using AWS S3: {config.S3_BUCKET}")
else:
    storage = get_storage_backend("local")
    print("⚠️  Using local storage (not recommended for production)")

# In-memory job store and queue
jobs: Dict[str, Dict[str, Any]] = {}
job_queue = queue.Queue()

# Worker infrastructure
worker_registry = WorkerRegistry(heartbeat_timeout=300)
worker_manager = WorkerManager(worker_registry)
job_dispatcher = JobDispatcher(worker_registry)

def worker_loop():
    """Background worker loop for local jobs."""
    print("Local worker thread started")
    while True:
        try:
            job_id = job_queue.get()
            if job_id is None:
                break
            
            job = jobs.get(job_id)
            if job:
                # Check if job should run locally or remotely
                if job.get("run_local", False):
                    updated_job = process_job(job)
                    jobs[job_id] = updated_job
                
            job_queue.task_done()
        except Exception as e:
            print(f"Worker error: {e}")

# Start local worker thread
worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()

def _create_job(job_type: str, payload: dict, run_local: bool = False):
    job_id = f"job_{job_type}_{uuid.uuid4().hex[:8]}"
    job = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "pending",
        "payload": payload,
        "created_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None,
        "run_local": run_local,
    }
    jobs[job_id] = job
    
    if run_local:
        job_queue.put(job_id)
    else:
        # Dispatch to remote worker
        _dispatch_to_remote_worker(job)
    
    return job

def _dispatch_to_remote_worker(job: Dict):
    """Dispatch job to remote GPU worker."""
    # Get available worker
    worker = worker_registry.get_available_worker(gpu_requirement=1)
    
    if not worker:
        # Provision new worker
        print(f"No workers available, provisioning for job {job['job_id']}")
        job["status"] = "provisioning_worker"
        # This would be async in production
        # For now, mark as pending
    else:
        # Dispatch to existing worker
        print(f"Dispatching job {job['job_id']} to worker {worker.worker_id}")
        job["status"] = "dispatched"
        # Async dispatch
        import asyncio
        asyncio.create_task(job_dispatcher.dispatch_to_worker(job, worker.worker_id))

@app.post("/audit")
async def audit(req: AuditRequest):
    # Audit runs locally (lightweight)
    job = _create_job("audit", req.dict(), run_local=True)
    return {"job_id": job["job_id"]}

@app.post("/distill")
async def distill(req: DistillRequest):
    # Distill can run remotely (GPU-intensive)
    job = _create_job("distill", req.dict(), run_local=False)
    return {"job_id": job["job_id"]}

@app.post("/train")
async def train(req: TrainRequest):
    # Train runs remotely (GPU-intensive)
    job = _create_job("train", req.dict(), run_local=False)
    return {"job_id": job["job_id"]}

@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    # Evaluate can run locally or remotely
    job = _create_job("evaluate", req.dict(), run_local=True)
    return {"job_id": job["job_id"]}

@app.post("/deploy")
async def deploy(req: DeployRequest):
    # Deploy runs locally (just starts inference server)
    job = _create_job("deploy", req.dict(), run_local=True)
    return {"job_id": job["job_id"]}

@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/logs/{job_id}/stream")
async def stream_logs(job_id: str):
    """Stream job logs in real-time."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        # Simple implementation: poll for logs
        last_position = 0
        while True:
            logs = await job_dispatcher.get_job_logs(job_id)
            if logs and len(logs) > last_position:
                new_logs = logs[last_position:]
                yield f"data: {new_logs}\n\n"
                last_position = len(logs)
            
            # Check if job is complete
            current_job = jobs.get(job_id)
            if current_job and current_job["status"] in ["completed", "failed"]:
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/workers")
async def list_workers():
    """List all registered workers."""
    workers = worker_registry.list_workers()
    return {"workers": [w.to_dict() for w in workers]}

@app.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str):
    """Worker heartbeat endpoint."""
    if worker_registry.heartbeat(worker_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Worker not found")

@app.post("/workers/register")
async def register_worker(req: WorkerRegistration):
    """
    Manually register a GPU worker with SSH credentials.
    
    This is used when you get SSH access from a GPU provider (e.g., Prime Intellect)
    and need to register it with the VPS control plane.
    """
    try:
        # Generate worker ID
        worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        
        # Register worker
        worker = worker_registry.register_worker(
            worker_id=worker_id,
            ssh_host=req.ssh_host,
            ssh_user=req.ssh_user,
            gpu_count=req.gpu_count,
            gpu_type=req.gpu_type,
            metadata={
                "provider": req.provider,
                **req.metadata
            }
        )
        
        # Bootstrap worker
        print(f"Bootstrapping worker {worker_id} at {req.ssh_host}")
        worker.status = WorkerStatus.BOOTSTRAPPING
        
        from src.infra.provisioning.ssh_bootstrap import SSHBootstrap
        
        ssh_key_path = Path(req.ssh_key_path) if req.ssh_key_path else config.SSH_KEY_PATH
        
        bootstrap = SSHBootstrap(
            ssh_host=req.ssh_host,
            ssh_user=req.ssh_user,
            ssh_key_path=ssh_key_path,
            repo_url=config.REPO_URL
        )
        
        # Run bootstrap in background (in production, use Celery/background tasks)
        import threading
        def bootstrap_worker():
            try:
                if bootstrap.bootstrap():
                    local_repo = Path(__file__).resolve().parents[2]
                    if bootstrap.sync_code(local_repo):
                        if bootstrap.install_dependencies():
                            worker.status = WorkerStatus.IDLE
                            print(f"✅ Worker {worker_id} ready")
                            return
                worker.status = WorkerStatus.ERROR
                print(f"❌ Worker {worker_id} bootstrap failed")
            except Exception as e:
                worker.status = WorkerStatus.ERROR
                print(f"❌ Worker {worker_id} error: {e}")
        
        threading.Thread(target=bootstrap_worker, daemon=True).start()
        
        return {
            "worker_id": worker_id,
            "status": "bootstrapping",
            "message": f"Worker {worker_id} registered and bootstrapping"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

