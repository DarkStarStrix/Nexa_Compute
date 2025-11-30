"""
Comprehensive integration test that verifies all components are wired correctly.

This test should ALWAYS pass if the system is properly set up.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SDK = ROOT / "sdk"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from nexa_compute.api.integration_check import check_integration


@pytest.mark.integration
def test_all_components_available():
    """CRITICAL: Test that all components are available and wired correctly.
    
    This test MUST pass for the system to be deployment-ready.
    """
    status = check_integration()
    
    # All components must be available
    assert status["storage_registry"] is True, \
        "Storage registry not available - check src/storage/registry.py"
    assert status["worker_processor"] is True, \
        "Worker processor not available - check src/workers/worker.py"
    assert status["server_config"] is True, \
        "Server config not available - check src/server/config.py"
    assert status["server_models"] is True, \
        "Server models not available - check src/server/models.py"
    assert status["all_available"] is True, \
        "Not all components are available - system is not deployment-ready"


@pytest.mark.integration
def test_component_imports_directly():
    """Test that components can be imported directly."""
    # Test storage registry
    from storage.registry import get_dataset_uri, get_checkpoint_uri
    assert callable(get_dataset_uri)
    assert callable(get_checkpoint_uri)
    
    # Test worker processor
    from workers.worker import process_job
    assert callable(process_job)
    
    # Test server config
    from server.config import Config
    assert Config is not None
    
    # Test server models
    from server.models import BaseJob
    assert BaseJob is not None


@pytest.mark.integration
def test_api_can_use_components():
    """Test that API can use all components."""
    # Test that API can import worker processor
    from nexa_compute.api.worker_agent import WorkerAgent
    agent = WorkerAgent()
    assert agent is not None
    
    # Test that API can use storage
    from nexa_compute.api.endpoints.artifacts import get_storage
    storage = get_storage()
    assert storage is not None
    
    # Test that API can use billing service
    from nexa_compute.api.services.billing_service import BillingService
    from nexa_compute.api.database import SessionLocal
    db = SessionLocal()
    billing = BillingService(db)
    assert billing is not None

