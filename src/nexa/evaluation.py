import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from nexa_eval.evaluate_distillation import run_distillation_evaluation

def run_evaluation(
    checkpoint_id: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run evaluation on a checkpoint.
    
    Args:
        checkpoint_id: Identifier for the checkpoint
        output_dir: Directory to store evaluation results
        
    Returns:
        Dict with evaluation metrics
    """
    # In a real scenario, we would load the checkpoint and generate outputs first.
    # For this MVP wrapper, we'll assume we are evaluating the distillation quality 
    # associated with the dataset that produced this checkpoint, or similar.
    # Since evaluate_distillation.py evaluates teacher inputs/outputs, let's map it to that.
    
    # Assuming checkpoint_id maps to a dataset_id for now, or we have a way to get the data.
    # Let's assume checkpoint_id is "ckpt_{dataset_id}"
    dataset_id = checkpoint_id.replace("ckpt_", "")
    
    distill_dir = ROOT / "artifacts" / "distill" / dataset_id
    inputs_path = distill_dir / "collected.parquet" # Teacher inputs/outputs are here
    outputs_path = distill_dir / "sft.parquet"      # Final SFT data
    
    base_dir = Path(output_dir) if output_dir else ROOT / "artifacts" / "evals" / checkpoint_id
    base_dir.mkdir(parents=True, exist_ok=True)
    report_path = base_dir / "report.json"
    
    if not inputs_path.exists() or not outputs_path.exists():
        raise FileNotFoundError(f"Data for evaluation not found in {distill_dir}")
        
    metrics = run_distillation_evaluation(
        inputs_path=inputs_path,
        outputs_path=outputs_path,
        output_report_path=report_path
    )
    
    return {
        "eval_id": f"eval_{checkpoint_id}",
        "scores": metrics,
        "report_uri": str(report_path)
    }
