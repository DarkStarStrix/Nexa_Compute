import logging
import traceback
from datetime import datetime
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.server.models import BaseJob
except ImportError:
    # Fallback if running from different context
    BaseJob = None

try:
    from src.nexa.data_quality import audit_dataset
except ImportError:
    audit_dataset = None

try:
    from src.nexa.distillation import run_distillation
except ImportError:
    run_distillation = None

try:
    from src.nexa.training import run_training
except ImportError:
    run_training = None

try:
    from src.nexa.evaluation import run_evaluation
except ImportError:
    run_evaluation = None

try:
    from src.nexa.deployment import deploy_model
except ImportError:
    deploy_model = None

logger = logging.getLogger(__name__)

def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a job based on its type.
    Returns the updated job dict.
    """
    job_id = job["job_id"]
    job_type = job["job_type"]
    payload = job.get("payload", {})
    
    logger.info(f"Processing job {job_id} ({job_type})")
    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat()
    
    try:
        result = None
        if job_type == "audit":
            if audit_dataset:
                result = audit_dataset(**payload)
            else:
                raise ImportError("audit_dataset not available")
        elif job_type == "distill":
            if run_distillation:
                result = run_distillation(**payload)
            else:
                raise ImportError("run_distillation not available")
        elif job_type == "train":
            if run_training:
                result = run_training(**payload)
            else:
                raise ImportError("run_training not available")
        elif job_type == "evaluate":
            if run_evaluation:
                result = run_evaluation(**payload)
            else:
                raise ImportError("run_evaluation not available")
        elif job_type == "deploy":
            if deploy_model:
                result = deploy_model(**payload)
            else:
                raise ImportError("deploy_model not available")
        elif job_type == "generate":
            # Data generation - placeholder
            result = {"message": "Data generation not yet implemented", "samples": 0}
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
