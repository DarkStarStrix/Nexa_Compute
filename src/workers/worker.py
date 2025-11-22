import logging
import traceback
from datetime import datetime
from typing import Dict, Any

from src.server.models import BaseJob
from src.nexa.data_quality import audit_dataset
from src.nexa.distillation import run_distillation
from src.nexa.training import run_training
from src.nexa.evaluation import run_evaluation
from src.nexa.deployment import deploy_model

logger = logging.getLogger(__name__)

def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a job based on its type.
    Returns the updated job dict.
    """
    job_id = job["job_id"]
    job_type = job["job_type"]
    payload = job["payload"]
    
    logger.info(f"Processing job {job_id} ({job_type})")
    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat()
    
    try:
        result = None
        if job_type == "audit":
            result = audit_dataset(**payload)
        elif job_type == "distill":
            result = run_distillation(**payload)
        elif job_type == "train":
            result = run_training(**payload)
        elif job_type == "evaluate":
            result = run_evaluation(**payload)
        elif job_type == "deploy":
            result = deploy_model(**payload)
        else:
            raise ValueError(f"Unknown job type: {job_type}")
            
        job["result"] = result
        job["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job["error"] = str(e)
        job["status"] = "failed"
        job["logs"] = traceback.format_exc()
        
    job["completed_at"] = datetime.utcnow().isoformat()
    return job
