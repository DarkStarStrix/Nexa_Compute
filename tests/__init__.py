"""
Test suite for Nexa Forge deployment verification.

Run all tests:
    pytest tests/ -v

Run specific test suites:
    pytest tests/test_api.py -v          # API endpoint tests
    pytest tests/test_sdk.py -v          # SDK client tests
    pytest tests/test_integration.py -v  # Integration tests
    pytest tests/test_deployment.py -v   # Deployment readiness tests

Run with markers:
    pytest -m api -v          # API tests only
    pytest -m sdk -v          # SDK tests only
    pytest -m integration -v # Integration tests only
    pytest -m deployment -v   # Deployment tests only
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SDK = ROOT / "sdk"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

# Test modules
__all__ = [
    "test_api",
    "test_sdk",
    "test_integration",
    "test_deployment",
    "test_infra",
    "test_config",
    "test_registry"
]

