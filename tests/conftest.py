import os
import re
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

# Disable OpenTelemetry tracing in test environment (no collector running)
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")

# Disable WandB in tests unless explicitly enabled
os.environ.setdefault("WANDB_MODE", "disabled")

# Encourage torch to use modern NVML bindings and silence legacy warning noise
# Set this BEFORE any torch imports
os.environ.setdefault("PYTORCH_NVML_BASE_MODULE", "nvidia-ml-py")

# Suppress pynvml deprecation warnings from PyTorch (it imports pynvml internally)
# This must be done before torch is imported anywhere
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch")

# Suppress OpenTelemetry connection errors (collector not running in tests)
warnings.filterwarnings("ignore", message=".*failed to connect.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Failed to export.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Transient error.*", category=RuntimeWarning)

# Track passed tests for README badge update
_passed_tests = 0


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "api: marks tests for API endpoints")
    config.addinivalue_line("markers", "sdk: marks tests for SDK client")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "deployment: marks deployment readiness tests")
    config.addinivalue_line("markers", "infra: marks infrastructure tests")


def pytest_runtest_logreport(report):
    """Track passed tests for README badge update."""
    global _passed_tests
    if report.when == "call" and report.outcome == "passed":
        _passed_tests += 1

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
    # Disable telemetry and external services in tests
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("WANDB_MODE", "disabled")


def pytest_sessionfinish(session, exitstatus):
    """Update README badge with test count after tests complete."""
    global _passed_tests
    if _passed_tests > 0:
        readme_path = ROOT / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            badge_pattern = r'(\[!\[Tests\]\(https://img\.shields\.io/badge/tests-)\d+(\%20passing-brightgreen\)\]\(tests/\))'
            replacement = rf'\g<1>{_passed_tests}\g<2>'
            updated_content = re.sub(badge_pattern, replacement, content)
            if updated_content != content:
                readme_path.write_text(updated_content)
