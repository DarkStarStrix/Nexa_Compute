"""
Configuration tests.
"""
from pathlib import Path
import yaml
import pytest

def test_load_baseline_config():
    """Test loading the baseline training configuration."""
    config_path = Path("nexa_train/configs/baseline.yaml")
    assert config_path.exists()
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    assert "model" in config
    assert "data" in config
    assert "training" in config

def test_compute_plan_validity():
    """Test that compute plans are valid YAML and contain required fields."""
    plan_path = Path("docs/compute_plans/v1_stability.yaml")
    
    with open(plan_path) as f:
        plan = yaml.safe_load(f)
        
    required_sections = ["run", "cluster", "data", "model", "training"]
    for section in required_sections:
        assert section in plan, f"Missing '{section}' in compute plan"
