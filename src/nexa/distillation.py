import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from nexa_distill.pipeline import DistillationPipeline

def run_distillation(
    dataset_id: str,
    teacher: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the distillation pipeline.
    
    Args:
        dataset_id: Identifier for the dataset (folder name in artifacts/datasets)
        teacher: Teacher model ID
        api_key: OpenAI API key
        output_dir: Directory to store results
        
    Returns:
        Dict with paths to distilled artifacts
    """
    # Setup paths
    base_dir = Path(output_dir) if output_dir else ROOT / "artifacts" / "distill" / dataset_id
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary config for this run
    config = {
        "defaults": {
            "prompt_column": "prompt",
            "context_column": "context",
            "task_type_column": "task_type"
        },
        "storage": {
            "raw_dataset": str(ROOT / "artifacts" / "datasets" / dataset_id / "raw.parquet"),
            "collected_dataset": str(base_dir / "collected.parquet"),
            "filtered_dataset": str(base_dir / "filtered.parquet"),
            "regen_dataset": str(base_dir / "regen.parquet"),
            "sft_jsonl": str(base_dir / "sft.jsonl"),
            "sft_parquet": str(base_dir / "sft.parquet")
        },
        "collection": {
            "teacher_model": teacher,
            "batch_size": 4
        }
    }
    
    config_path = base_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # Initialize pipeline
    pipeline = DistillationPipeline(config_path=config_path)
    
    # Run stages
    pipeline.collect_teacher(api_key=api_key)
    pipeline.filter_teacher()
    pipeline.package_sft()
    
    return {
        "distilled_dataset_id": dataset_id,
        "sft_dataset_uri": str(base_dir / "sft.parquet"),
        "config_uri": str(config_path)
    }
