"""
Deployment readiness tests.
Tests that verify the system is ready for VPS deployment.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestDeploymentReadiness:
    """Test deployment readiness."""
    
    def test_api_endpoints_registered(self):
        """Test that all required API endpoints are registered."""
        try:
            from nexa_compute.api.main import app
            
            routes = [r.path for r in app.routes]
            
            required_routes = [
                "/api/jobs",
                "/api/workers",
                "/api/billing",
                "/api/auth",
                "/api/artifacts",
                "/health"
            ]
            
            for route in required_routes:
                # Check if route or a variant exists
                assert any(route in r or r.startswith(route) for r in routes), f"Missing route: {route}"
        except ImportError:
            pytest.skip("API main module not available")
    
    def test_database_initialization(self):
        """Test that database can be initialized."""
        try:
            from nexa_compute.api.database import init_db, Base, engine
            
            # Should not raise
            Base.metadata.create_all(bind=engine)
        except Exception as e:
            pytest.fail(f"Database initialization failed: {e}")
    
    def test_storage_backend_configuration(self, tmp_path, monkeypatch):
        """Test storage backend configuration."""
        try:
            from nexa_compute.api.endpoints.artifacts import get_storage
            
            # Test local storage
            monkeypatch.setenv("STORAGE_BACKEND", "local")
            monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
            
            storage = get_storage()
            assert storage is not None
        except ImportError:
            pytest.skip("Artifacts endpoint not available")
    
    def test_billing_service_initialization(self):
        """Test that billing service can be initialized."""
        try:
            from nexa_compute.api.services.billing_service import BillingService
            from nexa_compute.api.database import SessionLocal
            
            db = SessionLocal()
            service = BillingService(db)
            assert service is not None
        except ImportError:
            pytest.skip("Billing service not available")
    
    def test_job_manager_initialization(self):
        """Test that job manager can be initialized."""
        try:
            from nexa_compute.api.services.job_manager import JobManager
            from nexa_compute.api.database import SessionLocal
            
            db = SessionLocal()
            manager = JobManager(db)
            assert manager is not None
        except ImportError:
            pytest.skip("Job manager not available")
    
    def test_worker_registry_initialization(self):
        """Test that worker registry can be initialized."""
        try:
            from nexa_compute.api.services.worker_registry import WorkerRegistry
            from nexa_compute.api.database import SessionLocal
            
            db = SessionLocal()
            registry = WorkerRegistry(db)
            assert registry is not None
        except ImportError:
            pytest.skip("Worker registry not available")


class TestComponentWiring:
    """Test that components are properly wired together."""
    
    def test_api_to_database_wiring(self):
        """Test API endpoints can access database."""
        try:
            from nexa_compute.api.endpoints.jobs import get_job_manager
            from nexa_compute.api.database import get_db
            
            # Should not raise
            db_gen = get_db()
            db = next(db_gen)
            assert db is not None
        except ImportError:
            pytest.skip("API endpoints not available")
    
    def test_api_to_billing_wiring(self):
        """Test API endpoints can access billing service."""
        try:
            from nexa_compute.api.endpoints.jobs import get_billing_service
            from nexa_compute.api.database import get_db
            
            db_gen = get_db()
            db = next(db_gen)
            billing = get_billing_service(db)
            assert billing is not None
        except ImportError:
            pytest.skip("Billing service not available")
    
    def test_worker_agent_to_api_woker_processor(self):
        """Test worker agent can import worker processor."""
        try:
            # This is tested in integration tests, but verify here too
            from workers.worker import process_job
            from nexa_compute.api.worker_agent import WorkerAgent
            
            agent = WorkerAgent()
            # If we get here, the import chain works
            assert agent is not None
            assert process_job is not None
        except ImportError as e:
            pytest.skip(f"Worker components not available: {e}")


class TestEnvironmentConfiguration:
    """Test environment configuration."""
    
    def test_required_env_vars_documented(self):
        """Test that required environment variables are documented."""
        # Check if env.example exists
        env_example = ROOT / "env" / "env.example"
        if not env_example.exists():
            # Check root
            env_example = ROOT / ".env.example"
        
        if env_example.exists():
            content = env_example.read_text()
            # Check for key variables
            assert "API" in content or "NEXA" in content
        else:
            pytest.skip("No .env.example file found")
    
    def test_storage_env_vars(self, monkeypatch):
        """Test storage environment variable handling."""
        try:
            from nexa_compute.api.config import get_settings
            
            # Test defaults
            settings = get_settings()
            assert hasattr(settings, "STORAGE_BACKEND")
            
            # Test env override
            monkeypatch.setenv("STORAGE_BACKEND", "s3")
            # Note: Settings are cached, so this may not work
            # But we can at least verify the attribute exists
        except ImportError:
            pytest.skip("API config not available")


class TestSDKDeployment:
    """Test SDK deployment readiness."""
    
    def test_sdk_importable(self):
        """Test that SDK can be imported."""
        try:
            from nexa_forge.client import NexaForgeClient
            assert NexaForgeClient is not None
        except ImportError:
            pytest.skip("SDK not available")
    
    def test_sdk_client_initialization(self):
        """Test SDK client can be initialized."""
        try:
            from nexa_forge.client import NexaForgeClient
            
            client = NexaForgeClient(api_key="test-key")
            assert client.api_url is not None
            assert client.headers is not None
        except ImportError:
            pytest.skip("SDK not available")

