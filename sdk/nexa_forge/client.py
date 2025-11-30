import os
import requests
from typing import Dict, Any, Optional, List, Union
from enum import Enum

class JobType(str, Enum):
    GENERATE = "generate"
    AUDIT = "audit"
    DISTILL = "distill"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"

class NexaForgeClient:
    """
    Python SDK for Nexa Forge.
    
    Usage:
        client = NexaForgeClient(api_key="your_api_key")
        job = client.generate(domain="biology", num_samples=100)
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEXA_API_KEY")
        raw_url = api_url or os.getenv("NEXA_API_URL", "http://localhost:8000/api")
        self.api_url = self._normalize_api_url(raw_url)
        
        if not self.api_key:
            print("Warning: No API Key provided. Some endpoints may fail.")
            
        self.headers = {
            "Content-Type": "application/json",
            "X-Nexa-Api-Key": self.api_key or ""
        }

    @staticmethod
    def _normalize_api_url(url: str) -> str:
        sanitized = url.rstrip("/")
        if not sanitized.endswith("/api"):
            sanitized = f"{sanitized}/api"
        return sanitized

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_url}/{endpoint}"
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.api_url}/{endpoint}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def submit_job(self, job_type: Union[str, JobType], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a new job."""
        if isinstance(job_type, JobType):
            job_type = job_type.value
            
        return self._post(f"jobs/{job_type}", {"payload": payload})

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status and details."""
        return self._get(f"jobs/{job_id}")

    def list_jobs(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent jobs."""
        endpoint = f"jobs/?limit={limit}"
        if status:
            endpoint += f"&status={status}"
        return self._get(endpoint)

    # --- Convenience Methods ---

    def generate(self, domain: str, num_samples: int, **kwargs) -> Dict[str, Any]:
        """
        Submit a data generation job.
        
        Args:
            domain: The domain to generate data for (e.g., "biology", "coding").
            num_samples: Number of samples to generate.
        """
        return self.submit_job("generate", {"domain": domain, "num_samples": num_samples, **kwargs})

    def audit(self, dataset_uri: str, **kwargs) -> Dict[str, Any]:
        """
        Submit a data audit job.
        
        Args:
            dataset_uri: URI of the dataset to audit (e.g., "s3://bucket/data.parquet").
        """
        return self.submit_job("audit", {"dataset_uri": dataset_uri, **kwargs})

    def distill(self, teacher_model: str, student_model: str, dataset_uri: str, **kwargs) -> Dict[str, Any]:
        """
        Submit a model distillation job.
        
        Args:
            teacher_model: Name/ID of the teacher model.
            student_model: Name/ID of the student model.
            dataset_uri: URI of the training dataset.
        """
        return self.submit_job("distill", {
            "teacher_model": teacher_model,
            "student_model": student_model,
            "dataset_uri": dataset_uri,
            **kwargs
        })

    def train(self, model_id: str, dataset_uri: str, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Submit a training job.
        
        Args:
            model_id: Base model to train.
            dataset_uri: Training dataset URI.
            epochs: Number of training epochs.
        """
        return self.submit_job("train", {
            "model_id": model_id,
            "dataset_uri": dataset_uri,
            "epochs": epochs,
            **kwargs
        })

    def evaluate(self, model_id: str, benchmark: str, **kwargs) -> Dict[str, Any]:
        """
        Submit an evaluation job.
        
        Args:
            model_id: Model to evaluate.
            benchmark: Benchmark dataset/suite to run.
        """
        return self.submit_job("evaluate", {
            "model_id": model_id,
            "benchmark": benchmark,
            **kwargs
        })

    def deploy(self, model_id: str, region: str = "us-east-1", **kwargs) -> Dict[str, Any]:
        """
        Deploy a model to an endpoint.
        
        Args:
            model_id: Model to deploy.
            region: Target region.
        """
        return self.submit_job("deploy", {
            "model_id": model_id,
            "region": region,
            **kwargs
        })
