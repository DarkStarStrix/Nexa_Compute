import os
import time
import socket
import requests
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

# Import the job processor logic
# Assuming src is in python path
try:
    from src.workers.worker import process_job
except ImportError:
    # Fallback if running from root
    from workers.worker import process_job

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = os.getenv("NEXA_API_URL", "http://localhost:8000/api")
WORKER_ID = os.getenv("WORKER_ID", f"worker-{uuid.uuid4().hex[:8]}")
HOSTNAME = socket.gethostname()
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

class WorkerAgent:
    def __init__(self):
        self.worker_id = WORKER_ID
        self.api_url = API_URL
        self.current_job_id: Optional[str] = None
        self.status = "idle"

    def register(self):
        """Register worker with the API."""
        logger.info(f"Registering worker {self.worker_id} at {self.api_url}")
        try:
            payload = {
                "worker_id": self.worker_id,
                "hostname": HOSTNAME,
                "gpu_count": 1,  # TODO: Detect GPU
                "gpu_type": "Generic" # TODO: Detect GPU
            }
            response = requests.post(f"{self.api_url}/workers/register", json=payload)
            response.raise_for_status()
            logger.info("Registration successful")
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            # We continue anyway, maybe it's already registered or API will accept heartbeat

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
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get next job: {e}")
            return None

    def update_job_status(self, job_id: str, status: str, result: Any = None, error: str = None):
        """Update job status."""
        try:
            payload = {
                "status": status,
                "result": result,
                "error": error
            }
            requests.post(f"{self.api_url}/jobs/{job_id}/status", json=payload)
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
                    logger.info(f"Received job {self.current_job_id}: {job['job_type']}")
                    
                    # Update status to running
                    self.update_job_status(self.current_job_id, "running")
                    
                    try:
                        # Execute job
                        # We adapt the job dict to what process_job expects if needed
                        # process_job expects: job_id, job_type, payload
                        processed_job = process_job(job)
                        
                        if processed_job.get("status") == "completed":
                            self.update_job_status(self.current_job_id, "completed", result=processed_job.get("result"))
                        else:
                            self.update_job_status(self.current_job_id, "failed", error=processed_job.get("error"))
                            
                    except Exception as e:
                        logger.error(f"Job execution failed: {e}")
                        self.update_job_status(self.current_job_id, "failed", error=str(e))
                    
                    self.status = "idle"
                    self.current_job_id = None
            
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    agent = WorkerAgent()
    agent.run()
