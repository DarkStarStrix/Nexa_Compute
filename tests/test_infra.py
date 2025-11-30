"""
Infrastructure tests for NexaCompute.
Tests core deployment, file system, and CLI capabilities.
"""

import os
import subprocess
import pytest
from pathlib import Path

@pytest.mark.infra
def test_directory_structure():
    """Verify critical directories exist."""
    roots = ["nexa_data", "nexa_train", "nexa_infra", "docs"]
    for root in roots:
        assert Path(root).exists(), f"Missing core module: {root}"

@pytest.mark.infra
def test_cli_entrypoint():
    """Test that the nexa CLI is installed and runnable."""
    result = subprocess.run(["nexa", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Nexa Compute control surface" in result.stdout

@pytest.mark.infra
def test_import_core_modules():
    """Test that core modules are importable in the python environment."""
    try:
        import nexa_data
        import nexa_train
        import nexa_infra
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")

@pytest.mark.infra
def test_terraform_files_exist():
    """Verify IaC definitions are present."""
    assert Path("nexa_infra/terraform/main.tf").exists()
    assert Path("nexa_infra/terraform/variables.tf").exists()

@pytest.mark.infra
def test_compute_plans_exist():
    """Verify compute plan templates are present."""
    plans_dir = Path("docs/compute_plans")
    assert plans_dir.exists()
    assert (plans_dir / "v1_stability.yaml").exists()
    assert (plans_dir / "v2_performance.yaml").exists()

