"""
Tests for Nexa Forge SDK client.
Tests SDK functionality and API integration.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src and sdk to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SDK = ROOT / "sdk"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from nexa_forge.client import NexaForgeClient, JobType


class TestSDKClient:
    """Test SDK client initialization and basic functionality."""
    
    def test_client_init_default(self):
        """Test client initialization with defaults."""
        client = NexaForgeClient()
        # API URL should include /api suffix (default from env or hardcoded)
        assert client.api_url.endswith("/api") or client.api_url == "http://localhost:8000/api"
        assert "Content-Type" in client.headers
    
    def test_client_init_custom(self):
        """Test client initialization with custom values."""
        client = NexaForgeClient(
            api_key="test-key",
            api_url="http://test.com/api"
        )
        assert client.api_url == "http://test.com/api"
        assert client.headers["X-Nexa-Api-Key"] == "test-key"
    
    def test_client_init_env_var(self, monkeypatch):
        """Test client initialization from environment variables."""
        monkeypatch.setenv("NEXA_API_KEY", "env-key")
        monkeypatch.setenv("NEXA_API_URL", "http://env.com/api")
        
        client = NexaForgeClient()
        assert client.api_url == "http://env.com/api"
        assert client.headers["X-Nexa-Api-Key"] == "env-key"


class TestSDKJobMethods:
    """Test SDK job submission methods."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP methods."""
        client = NexaForgeClient(api_key="test-key")
        client._post = Mock()
        client._get = Mock()
        return client
    
    def test_submit_job(self, mock_client):
        """Test submit_job method."""
        mock_client._post.return_value = {"job_id": "test-123", "status": "pending"}
        
        result = mock_client.submit_job("generate", {"domain": "test"})
        
        assert result["job_id"] == "test-123"
        mock_client._post.assert_called_once()
        call_args = mock_client._post.call_args
        assert "jobs/generate" in call_args[0][0]
    
    def test_submit_job_with_enum(self, mock_client):
        """Test submit_job with JobType enum."""
        mock_client._post.return_value = {"job_id": "test-123"}
        
        result = mock_client.submit_job(JobType.GENERATE, {"domain": "test"})
        
        assert result["job_id"] == "test-123"
        call_args = mock_client._post.call_args
        assert "jobs/generate" in call_args[0][0]
    
    def test_get_job(self, mock_client):
        """Test get_job method."""
        mock_client._get.return_value = {"job_id": "test-123", "status": "completed"}
        
        result = mock_client.get_job("test-123")
        
        assert result["job_id"] == "test-123"
        mock_client._get.assert_called_once_with("jobs/test-123")
    
    def test_list_jobs(self, mock_client):
        """Test list_jobs method."""
        mock_client._get.return_value = [{"job_id": "test-123"}]
        
        result = mock_client.list_jobs(limit=10, status="completed")
        
        assert len(result) == 1
        mock_client._get.assert_called_once()
        call_args = mock_client._get.call_args
        assert "limit=10" in call_args[0][0]
        assert "status=completed" in call_args[0][0]


class TestSDKConvenienceMethods:
    """Test SDK convenience methods for different job types."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked submit_job."""
        client = NexaForgeClient(api_key="test-key")
        client.submit_job = Mock()
        return client
    
    def test_generate(self, mock_client):
        """Test generate convenience method."""
        mock_client.submit_job.return_value = {"job_id": "test-123"}
        
        result = mock_client.generate("biology", 100)
        
        assert result["job_id"] == "test-123"
        mock_client.submit_job.assert_called_once_with(
            "generate",
            {"domain": "biology", "num_samples": 100}
        )
    
    def test_audit(self, mock_client):
        """Test audit convenience method."""
        mock_client.submit_job.return_value = {"job_id": "test-123"}
        
        result = mock_client.audit("s3://bucket/data.parquet")
        
        mock_client.submit_job.assert_called_once_with(
            "audit",
            {"dataset_uri": "s3://bucket/data.parquet"}
        )
    
    def test_distill(self, mock_client):
        """Test distill convenience method."""
        mock_client.submit_job.return_value = {"job_id": "test-123"}
        
        result = mock_client.distill("teacher", "student", "s3://data")
        
        call_args = mock_client.submit_job.call_args
        assert call_args[0][0] == "distill"
        payload = call_args[0][1]
        assert payload["teacher_model"] == "teacher"
        assert payload["student_model"] == "student"
    
    def test_train(self, mock_client):
        """Test train convenience method."""
        mock_client.submit_job.return_value = {"job_id": "test-123"}
        
        result = mock_client.train("model-1", "s3://data", epochs=3)
        
        call_args = mock_client.submit_job.call_args
        assert call_args[0][0] == "train"
        payload = call_args[0][1]
        assert payload["epochs"] == 3
    
    def test_evaluate(self, mock_client):
        """Test evaluate convenience method."""
        mock_client.submit_job.return_value = {"job_id": "test-123"}
        
        result = mock_client.evaluate("model-1", "benchmark-1")
        
        # Verify submit_job was called
        assert mock_client.submit_job.called
        call_args = mock_client.submit_job.call_args
        # Check that it was called with evaluate
        assert call_args[0][0] == "evaluate"
        # Payload is the second positional argument
        assert "benchmark" in call_args[0][1]
    
    def test_deploy(self, mock_client):
        """Test deploy convenience method."""
        mock_client.submit_job.return_value = {"job_id": "test-123"}
        
        result = mock_client.deploy("model-1", region="us-west-2")
        
        # Verify submit_job was called
        assert mock_client.submit_job.called
        call_args = mock_client.submit_job.call_args
        # Check that it was called with deploy
        assert call_args[0][0] == "deploy"
        # Payload is the second positional argument
        assert "region" in call_args[0][1]

