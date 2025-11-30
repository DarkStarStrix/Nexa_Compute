import os
import time
import socket
import requests
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the job processor logic
try:
    from src.workers.worker import process_job
except ImportError:
    # Fallback if running from root
    try:
        from workers.worker import process_job
    except ImportError:
        # Last resort - try relative import
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from workers.worker import process_job

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = os.getenv("NEXA_API_URL", "http://localhost:8000/api")
WORKER_ID = os.getenv("WORKER_ID", f"worker-{uuid.uuid4().hex[:8]}")
HOSTNAME = socket.gethostname()
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

# Detect GPU count (simple detection)
def detect_gpu_count() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 1

def detect_gpu_type() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.name
    except ImportError:
        pass
    return "Generic"

class WorkerAgent:
    def __init__(self):
        self.worker_id = WORKER_ID
        self.api_url = API_URL
        self.current_job_id: Optional[str] = None
        self.status = "idle"
        self.job_start_time: Optional[float] = None
        self.gpu_count = detect_gpu_count()
        self.gpu_type = detect_gpu_type()
        self.job_logs: list[str] = []

    def register(self):
        """Register worker with the API."""
        logger.info(f"Registering worker {self.worker_id} at {self.api_url}")
        try:
            payload = {
                "worker_id": self.worker_id,
                "hostname": HOSTNAME,
                "gpu_count": self.gpu_count,
                "gpu_type": self.gpu_type
            }
            response = requests.post(f"{self.api_url}/workers/register", json=payload)
            response.raise_for_status()
            logger.info("Registration successful")
        except Exception as e:
            logger.error(f"Registration failed: {e}")

    def heartbeat(self):
        """Send heartbeat to API."""
        try:
            payload = {
                "worker_id": self.worker_id,
                "status": self.status,
                "current_job_id": self.current_job_id
            }
            requests.post(f"{self.api_url}/workers/heartbeat", json=payload)
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")

    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Poll for next job."""
        try:
            response = requests.post(f"{self.api_url}/workers/next_job", json={"worker_id": self.worker_id})
            if response.status_code == 200:
                data = response.json()
                return data if data else None
            return None
        except Exception as e:
            logger.error(f"Failed to get next job: {e}")
            return None

    def log(self, message: str):
        """Add a log message."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.job_logs.append(log_entry)
        logger.info(log_entry)

    def update_job_status(self, job_id: str, status: str, result: Any = None, error: str = None, gpu_hours: Optional[float] = None):
        """Update job status and record billing."""
        try:
            payload = {
                "status": status,
                "result": result,
                "error": error
            }
            if gpu_hours is not None:
                payload["gpu_hours"] = gpu_hours
                payload["gpu_count"] = self.gpu_count
            
            # Include logs if available
            if self.job_logs:
                payload["logs"] = "\n".join(self.job_logs)
            
            response = requests.post(f"{self.api_url}/jobs/{job_id}/status", json=payload)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    def run(self):
        self.register()
        
        while True:
            self.heartbeat()
            
            if self.status == "idle":
                job = self.get_next_job()
                if job:
                    self.status = "busy"
                    self.current_job_id = job["job_id"]
                    self.job_start_time = time.time()
                    self.job_logs = []
                    self.log(f"Starting job {self.current_job_id} ({job['job_type']})")
                    
                    # Update status to running
                    self.update_job_status(self.current_job_id, "running")
                    
                    try:
                        # Execute job
                        self.log("Executing job...")
                        processed_job = process_job(job)
                        
                        # Calculate GPU hours
                        job_duration_seconds = time.time() - self.job_start_time
                        gpu_hours = job_duration_seconds / 3600.0
                        
                        if processed_job.get("status") == "completed":
                            self.log(f"Job completed successfully in {gpu_hours:.2f} GPU hours")
                            self.update_job_status(
                                self.current_job_id, 
                                "completed", 
                                result=processed_job.get("result"),
                                gpu_hours=gpu_hours
                            )
                            logger.info(f"Job {self.current_job_id} completed in {gpu_hours:.2f} GPU hours")
                        else:
                            self.log(f"Job failed: {processed_job.get('error', 'Unknown error')}")
                            self.update_job_status(
                                self.current_job_id, 
                                "failed", 
                                error=processed_job.get("error"),
                                gpu_hours=gpu_hours
                            )
                            
                    except Exception as e:
                        logger.error(f"Job execution failed: {e}")
                        job_duration_seconds = time.time() - self.job_start_time if self.job_start_time else 0
                        gpu_hours = job_duration_seconds / 3600.0
                        self.log(f"Job execution exception: {str(e)}")
                        self.update_job_status(
                            self.current_job_id, 
                            "failed", 
                            error=str(e),
                            gpu_hours=gpu_hours
                        )
                    
                    self.status = "idle"
                    self.current_job_id = None
                    self.job_start_time = None
                    self.job_logs = []
            
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    agent = WorkerAgent()
    agent.run()
