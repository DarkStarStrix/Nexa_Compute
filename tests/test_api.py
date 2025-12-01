"""Unit tests for the FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from nexa_compute.api.main import app
from nexa_compute.api.auth import get_api_key
from nexa_compute.api.database import UserDB

# Override auth for all tests in this module
def override_get_api_key():
    user = UserDB(user_id="test-user", email="test@nexa.run", is_active=True)
    user._current_api_key_hash = "test-hash"
    user._current_api_key_id = "test-id"
    return user

app.dependency_overrides[get_api_key] = override_get_api_key

@pytest.fixture
def client():
    return TestClient(app)


class TestHealth:
    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        # The main app includes health.router which has /health (returns "status": "ok")
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestJobs:
    def test_create_generate_job(self, client):
        """Test creating a generate job."""
        response = client.post(
            "/api/jobs/generate",
            json={"payload": {"domain": "test", "num_samples": 10}}
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_create_train_job(self, client):
        """Test creating a train job."""
        response = client.post(
            "/api/jobs/train",
            json={"payload": {"model_id": "test-model", "dataset_uri": "test://data"}}
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_list_jobs(self, client):
        """Test listing jobs."""
        # Create a job first
        client.post("/api/jobs/generate", json={"payload": {"domain": "test", "num_samples": 10}})

        response = client.get("/api/jobs?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_job(self, client):
        """Test getting a specific job."""
        # Create a job
        create_response = client.post(
            "/api/jobs/generate",
            json={"payload": {"domain": "test", "num_samples": 10}}
        )
        job_id = create_response.json()["job_id"]

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

        # jobs.py uses POST for status update
        response = client.post(
            f"/api/jobs/{job_id}/status",
            json={"status": "running"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["job"]["status"] == "running"
