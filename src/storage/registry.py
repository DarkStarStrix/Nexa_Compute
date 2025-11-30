from pathlib import Path
from typing import Optional, Dict, Any
import json
import os

# Use environment variable or default to artifacts directory
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))

def get_dataset_uri(dataset_id: str) -> str:
    """Get URI for a dataset."""
    # Check datasets dir
    path = ARTIFACTS_DIR / "datasets" / dataset_id
    if path.exists():
        return str(path.absolute())
    # Check distill dir
    path = ARTIFACTS_DIR / "distill" / dataset_id
    if path.exists():
        return str(path.absolute())
    raise FileNotFoundError(f"Dataset {dataset_id} not found")

def get_checkpoint_uri(checkpoint_id: str) -> str:
    """Get URI for a checkpoint."""
    path = ARTIFACTS_DIR / "checkpoints" / checkpoint_id
    if path.exists():
        return str(path.absolute())
    raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")

def get_eval_uri(eval_id: str) -> str:
    """Get URI for an evaluation."""
    path = ARTIFACTS_DIR / "evals" / eval_id
    if path.exists():
        return str(path.absolute())
    raise FileNotFoundError(f"Evaluation {eval_id} not found")

def get_deployment_info(deployment_id: str) -> Dict[str, Any]:
    """Get info for a deployment."""
    path = ARTIFACTS_DIR / "deployments" / deployment_id
    if path.exists():
        manifest = path / "manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                return json.load(f)
    raise FileNotFoundError(f"Deployment {deployment_id} not found")
