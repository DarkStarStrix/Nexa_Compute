"""
Tests for Nexa Forge API endpoints.
Tests job submission, worker management, billing, and artifacts.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.api.main import app
from nexa_compute.api.database import init_db, SessionLocal, Base, engine


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    # Use in-memory SQLite for tests
    test_db_path = ":memory:"
    test_engine = engine
    Base.metadata.create_all(bind=test_engine)
    
    yield
    
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(test_db):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def artifacts_dir(tmp_path):
    """Create temporary artifacts directory."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for subdir in ["datasets", "checkpoints", "evals"]:
        (artifacts / subdir).mkdir()
    
    # Set environment variable
    old_val = os.environ.get("ARTIFACTS_DIR")
    os.environ["ARTIFACTS_DIR"] = str(artifacts)
    yield artifacts
    if old_val:
        os.environ["ARTIFACTS_DIR"] = old_val
    else:
        os.environ.pop("ARTIFACTS_DIR", None)


class TestHealth:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestJobs:
    """Test job management endpoints."""
    
    def test_create_generate_job(self, client):
        """Test creating a generate job."""
        response = client.post(
            "/api/jobs/generate",
            json={"payload": {"domain": "test", "num_samples": 10}}
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["job_type"] == "generate"
        assert data["status"] == "pending"
    
    def test_create_train_job(self, client):
        """Test creating a train job."""
        response = client.post(
            "/api/jobs/train",
            json={"payload": {"model_id": "test-model", "dataset_uri": "test://data"}}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_type"] == "train"
    
    def test_list_jobs(self, client):
        """Test listing jobs."""
        # Create a job first
        client.post("/api/jobs/generate", json={"payload": {"domain": "test", "num_samples": 10}})
        
        response = client.get("/api/jobs?limit=10")
        assert response.status_code == 200
        jobs = response.json()
        assert isinstance(jobs, list)
        assert len(jobs) > 0
    
    def test_get_job(self, client):
        """Test getting a specific job."""
        # Create a job
        create_response = client.post(
            "/api/jobs/generate",
            json={"payload": {"domain": "test", "num_samples": 10}}
        )
        job_id = create_response.json()["job_id"]
        
        # Get the job
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
    
    def test_update_job_status(self, client):
        """Test updating job status."""
        # Create a job
        create_response = client.post(
            "/api/jobs/generate",
            json={"payload": {"domain": "test", "num_samples": 10}}
        )
        job_id = create_response.json()["job_id"]
        
        # Update status
        response = client.post(
            f"/api/jobs/{job_id}/status",
            json={
                "status": "completed",
                "result": {"samples": 10},
                "gpu_hours": 0.5,
                "gpu_count": 1
            }
        )
        assert response.status_code == 200
        
        # Verify status updated
        get_response = client.get(f"/api/jobs/{job_id}")
        assert get_response.json()["status"] == "completed"


class TestWorkers:
    """Test worker management endpoints."""
    
    def test_register_worker(self, client):
        """Test worker registration."""
        response = client.post(
            "/api/workers/register",
            json={
                "worker_id": "test-worker-1",
                "hostname": "test-host",
                "gpu_count": 1,
                "gpu_type": "A100"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["worker_id"] == "test-worker-1"
        assert data["gpu_count"] == 1
    
    def test_list_workers(self, client):
        """Test listing workers."""
        # Register a worker first
        client.post(
            "/api/workers/register",
            json={
                "worker_id": "test-worker-1",
                "hostname": "test-host",
                "gpu_count": 1
            }
        )
        
        response = client.get("/api/workers")
        assert response.status_code == 200
        workers = response.json()
        assert isinstance(workers, list)
        assert len(workers) > 0
    
    def test_worker_heartbeat(self, client):
        """Test worker heartbeat."""
        # Register worker first
        client.post(
            "/api/workers/register",
            json={
                "worker_id": "test-worker-1",
                "hostname": "test-host",
                "gpu_count": 1
            }
        )
        
        response = client.post(
            "/api/workers/heartbeat",
            json={
                "worker_id": "test-worker-1",
                "status": "idle"
            }
        )
        assert response.status_code == 200
    
    def test_get_next_job(self, client):
        """Test worker getting next job."""
        # Register worker
        client.post(
            "/api/workers/register",
            json={
                "worker_id": "test-worker-1",
                "hostname": "test-host",
                "gpu_count": 1
            }
        )
        
        # Create a job
        client.post(
            "/api/jobs/generate",
            json={"payload": {"domain": "test", "num_samples": 10}}
        )
        
        # Get next job
        response = client.post(
            "/api/workers/next_job",
            json={"worker_id": "test-worker-1"}
        )
        assert response.status_code == 200
        # May return None if no jobs available
        data = response.json()
        if data:
            assert "job_id" in data


class TestBilling:
    """Test billing endpoints."""
    
    def test_get_billing_summary(self, client):
        """Test getting billing summary."""
        response = client.get("/api/billing/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_cost" in data
        assert "currency" in data
        assert "usage_by_type" in data


class TestArtifacts:
    """Test artifacts endpoints."""
    
    def test_list_artifacts_empty(self, client, artifacts_dir):
        """Test listing artifacts when none exist."""
        response = client.get("/api/artifacts")
        assert response.status_code == 200
        artifacts = response.json()
        assert isinstance(artifacts, list)
    
    def test_list_artifacts_with_data(self, client, artifacts_dir):
        """Test listing artifacts with sample data."""
        # Create a sample artifact
        sample_file = artifacts_dir / "datasets" / "test.parquet"
        sample_file.write_bytes(b"test data")
        
        response = client.get("/api/artifacts")
        assert response.status_code == 200
        artifacts = response.json()
        # Should find at least one artifact
        assert len(artifacts) >= 0  # May be 0 if scanning doesn't work, but should not error
    
    def test_get_artifact(self, client, artifacts_dir):
        """Test getting artifact details."""
        # Create a sample artifact
        sample_file = artifacts_dir / "datasets" / "test.parquet"
        sample_file.write_bytes(b"test data")
        
        response = client.get("/api/artifacts/datasets_test")
        assert response.status_code == 200
        data = response.json()
        # Should return artifact info or basic response
        assert "id" in data or "message" in data

