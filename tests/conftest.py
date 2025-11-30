import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SDK = ROOT / "sdk"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

# Encourage torch to use modern NVML bindings and silence legacy warning noise
os.environ.setdefault("PYTORCH_NVML_BASE_MODULE", "nvidia-ml-py")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
warnings.filterwarnings("ignore", message="The pynvml package is deprecated", category=FutureWarning)

# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "api: marks tests for API endpoints")
    config.addinivalue_line("markers", "sdk: marks tests for SDK client")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "deployment: marks deployment readiness tests")
    config.addinivalue_line("markers", "infra: marks infrastructure tests")

@pytest.fixture(scope="session")
def test_artifacts_dir(tmp_path_factory):
    """Create a temporary artifacts directory for testing."""
    return tmp_path_factory.mktemp("artifacts")

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, test_artifacts_dir):
    """Set up test environment variables."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(test_artifacts_dir))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("NEXA_API_URL", "http://localhost:8000")
