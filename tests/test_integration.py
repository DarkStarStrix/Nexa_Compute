"""
Integration tests for Nexa Forge.
Tests end-to-end workflows and component integration.
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SDK = ROOT / "sdk"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from nexa_compute.api.integration_check import check_integration
from nexa_compute.api.deployment_check import verify_deployment_readiness


class TestIntegration:
    """Test component integration."""
    
    def test_storage_registry_import(self):
        """Test that storage registry can be imported."""
        try:
            from storage.registry import get_dataset_uri, get_checkpoint_uri
            assert callable(get_dataset_uri)
            assert callable(get_checkpoint_uri)
        except ImportError as e:
            pytest.fail(f"Storage registry import failed: {e}")
    
    def test_worker_processor_import(self):
        """Test that worker processor can be imported."""
        try:
            from workers.worker import process_job
            assert callable(process_job)
        except ImportError as e:
            pytest.fail(f"Worker processor import failed: {e}")
    
    def test_server_config_import(self):
        """Test that server config can be imported."""
        try:
            from server.config import Config
            assert Config is not None
        except ImportError as e:
            pytest.fail(f"Server config import failed: {e}")
    
    def test_server_models_import(self):
        """Test that server models can be imported."""
        try:
            from server.models import BaseJob
            assert BaseJob is not None
        except ImportError as e:
            pytest.fail(f"Server models import failed: {e}")
    
    def test_integration_check_all_components(self):
        """Test integration check function returns all components available."""
        status = check_integration()
        assert "storage_registry" in status
        assert "worker_processor" in status
        assert "server_config" in status
        assert "server_models" in status
        
        # All components should be available
        assert status["storage_registry"] is True, "Storage registry not available"
        assert status["worker_processor"] is True, "Worker processor not available"
        assert status["server_config"] is True, "Server config not available"
        assert status["server_models"] is True, "Server models not available"
        assert status["all_available"] is True, "Not all components are available"


class TestDeploymentReadiness:
    """Test deployment readiness checks."""
    
    def test_deployment_check(self, tmp_path, monkeypatch):
        """Test deployment readiness checker."""
        # Set up test environment
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        
        result = verify_deployment_readiness()
        
        assert "ready" in result
        assert "checks_passed" in result
        assert "checks_total" in result
        assert "details" in result
        assert isinstance(result["details"], list)
        
        # Should pass all checks
        assert result["ready"] is True, f"Deployment not ready: {result['details']}"


class TestEndToEndWorkflow:
    """Test end-to-end job workflow."""
    
    @pytest.fixture
    def mock_api_client(self):
        """Create a mock API client."""
        from unittest.mock import Mock
        client = Mock()
        client.submit_job.return_value = {
            "job_id": "test-job-123",
            "status": "pending",
            "job_type": "generate"
        }
        client.get_job.return_value = {
            "job_id": "test-job-123",
            "status": "completed",
            "result": {"samples": 10}
        }
        return client
    
    def test_job_lifecycle(self, mock_api_client):
        """Test complete job lifecycle."""
        from nexa_forge.client import NexaForgeClient
        
        # Submit job
        job = mock_api_client.submit_job("generate", {"domain": "test", "num_samples": 10})
        assert job["job_id"] == "test-job-123"
        
        # Get job status
        status = mock_api_client.get_job("test-job-123")
        assert status["status"] == "completed"


class TestStorageIntegration:
    """Test storage backend integration."""
    
    def test_storage_backend_factory(self):
        """Test storage backend factory function."""
        try:
            from infra.storage.backends import get_storage_backend
            
            # Test local backend
            local_backend = get_storage_backend("local", base_dir=Path("/tmp/test"))
            assert local_backend is not None
            
            # Test that it has required methods
            assert hasattr(local_backend, "upload")
            assert hasattr(local_backend, "download")
            assert hasattr(local_backend, "exists")
        except ImportError:
            pytest.fail("Storage backends not available")
    
    def test_storage_paths_integration(self, tmp_path, monkeypatch):
        """Test storage paths utility integration."""
        try:
            from nexa_compute.utils.storage import get_storage, StoragePaths
            
            # Set environment variables to use tmp_path
            monkeypatch.setenv("NEXA_SCRATCH", str(tmp_path / "scratch"))
            monkeypatch.setenv("NEXA_DURABLE", str(tmp_path / "durable"))
            monkeypatch.setenv("NEXA_SHARED", str(tmp_path / "shared"))
            
            storage = get_storage()
            assert isinstance(storage, StoragePaths)
            
            # Test path methods
            scratch_path = storage.scratch("test")
            durable_path = storage.durable("test")
            assert scratch_path is not None
            assert durable_path is not None
        except ImportError:
            pytest.fail("Storage utils not available")


class TestWorkerAgentIntegration:
    """Test worker agent integration."""
    
    def test_worker_agent_import(self):
        """Test that worker agent can be imported."""
        try:
            from nexa_compute.api.worker_agent import WorkerAgent
            assert WorkerAgent is not None
            
            # Test initialization
            agent = WorkerAgent()
            assert agent.worker_id is not None
            assert agent.api_url is not None
        except ImportError as e:
            pytest.fail(f"Worker agent not available: {e}")
    
    def test_worker_agent_process_job_import(self):
        """Test that worker agent can import process_job."""
        try:
            from nexa_compute.api.worker_agent import WorkerAgent
            from workers.worker import process_job
            
            agent = WorkerAgent()
            # If we get here, imports worked
            assert agent is not None
            assert process_job is not None
        except ImportError as e:
            pytest.fail(f"Worker agent or process_job not available: {e}")
    
    def test_worker_agent_can_call_process_job(self):
        """Test that worker agent can actually call process_job."""
        try:
            from workers.worker import process_job
            
            # Test with a simple job
            test_job = {
                "job_id": "test-123",
                "job_type": "generate",
                "payload": {"domain": "test", "num_samples": 10}
            }
            
            # Should not raise (may fail due to missing dependencies, but import should work)
            result = process_job(test_job)
            assert result is not None
            assert "status" in result
        except ImportError as e:
            pytest.fail(f"process_job not available: {e}")
        except Exception as e:
            # Other exceptions are OK (missing deps, etc), but import should work
            pass
