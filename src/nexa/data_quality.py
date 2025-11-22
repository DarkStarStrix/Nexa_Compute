import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# Add project root to sys.path to import scripts
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from scripts.rank_data_quality import rank_data, RankerSpec, OpenRouterConfig
except ImportError:
    # Fallback if scripts cannot be imported directly
    raise ImportError("Could not import rank_data_quality from scripts. Ensure project root is in python path.")

def audit_dataset(
    dataset_uri: str,
    model: str = "openai/gpt-4o-mini",
    max_samples: Optional[int] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run data quality audit on a dataset.
    
    Args:
        dataset_uri: Path or URI to the dataset (parquet/jsonl)
        model: LLM model ID to use for judging
        max_samples: Limit number of samples to audit
        api_key: Optional API key for the model provider
        
    Returns:
        Dict containing metrics and path to scored dataset
    """
    input_path = Path(dataset_uri)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_uri}")
        
    # Load data
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".jsonl":
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Unsupported format. Use .parquet or .jsonl")
        
    if max_samples:
        df = df.head(max_samples)
        
    # Configure ranker
    ranker = RankerSpec(model_id=model)
    client_config = OpenRouterConfig(model=model, api_key=api_key)
    
    # Run ranking
    ranked_df = rank_data(
        df,
        ranker=ranker,
        client_config=client_config,
        dry_run=False
    )
    
    # Calculate metrics
    metrics = {
        "clarity_mean": float(ranked_df["clarity"].mean()),
        "correctness_mean": float(ranked_df["correctness"].mean()),
        "educational_mean": float(ranked_df["educational_value"].mean()),
        "sample_count": len(ranked_df)
    }
    
    # Save result (in a real system, this would go to artifact storage)
    output_path = input_path.parent / f"{input_path.stem}_scored.parquet"
    ranked_df.to_parquet(output_path, index=False)
    
    return {
        "metrics": metrics,
        "scored_dataset_uri": str(output_path)
    }
