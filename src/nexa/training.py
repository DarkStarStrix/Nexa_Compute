import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from nexa_train.train import run_training_job

def run_training(
    dataset_id: str,
    model: str = "Mistral-7B",
    epochs: int = 3,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a training job.
    
    Args:
        dataset_id: Identifier for the dataset (folder name in artifacts/distill)
        model: Base model to fine-tune
        epochs: Number of training epochs
        output_dir: Directory to store checkpoints
        
    Returns:
        Dict with checkpoint info
    """
    # Setup paths
    base_dir = Path(output_dir) if output_dir else ROOT / "artifacts" / "checkpoints" / f"{dataset_id}_{model}"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = {
        "project": "nexa-train",
        "run_name": f"train-{dataset_id}",
        "output_dir": str(base_dir),
        "model": {
            "name": model,
            "parameters": {
                "pretrained": True
            }
        },
        "data": {
            "train_path": str(ROOT / "artifacts" / "distill" / dataset_id / "sft.parquet")
        },
        "training": {
            "epochs": epochs,
            "batch_size": 4,
            "learning_rate": 2e-5
        }
    }
    
    config_path = base_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # Run training
    artifact_meta = run_training_job(config_path)
    
    return {
        "checkpoint_id": artifact_meta.hash,
        "checkpoint_uri": artifact_meta.uri,
        "metrics": artifact_meta.labels
    }
