"""Utility for querying processed datasets with consistent paths."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class DataQuery:
    """Query interface for organized processed data."""
    
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else Path(__file__).parent.parent.parent
        self.processed_root = self.root / "data" / "processed"
    
    def get_teacher_inputs(self, version: str = "v1") -> pd.DataFrame:
        """Load teacher input dataset."""
        path = self.processed_root / "distillation" / "teacher_inputs" / f"teacher_inputs_{version}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Teacher inputs not found: {path}")
        return pd.read_parquet(path)
    
    def get_training_split(self, split: str = "train", version: str = "v1") -> pd.DataFrame:
        """Load training/validation/test split."""
        path = self.processed_root / "training" / split / f"{split}_{version}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Training split not found: {path}")
        return pd.read_parquet(path)
    
    def get_pretrain_dataset(self, shard: str = "001") -> pd.DataFrame:
        """Load pretrain dataset shard (JSONL format)."""
        import json
        path = self.processed_root / "training" / "pretrain" / f"{shard}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Pretrain shard not found: {path}")
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    
    def get_evaluation_predictions(self, run_id: str) -> pd.DataFrame:
        """Load evaluation predictions for a run."""
        path = self.processed_root / "evaluation" / "predictions" / f"predictions_{run_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Predictions not found: {path}")
        return pd.read_parquet(path)
    
    def list_available_datasets(self) -> dict:
        """List all available processed datasets."""
        datasets = {}
        for category in ["distillation", "training", "evaluation"]:
            category_dir = self.processed_root / category
            if category_dir.exists():
                parquet_files = [f.name for f in category_dir.rglob("*.parquet")]
                jsonl_files = [f.name for f in category_dir.rglob("*.jsonl")]
                datasets[category] = parquet_files + jsonl_files
        return datasets


def query_teacher_inputs(version: str = "v1") -> pd.DataFrame:
    """Convenience function to load teacher inputs."""
    return DataQuery().get_teacher_inputs(version)


if __name__ == "__main__":
    # Example usage
    query = DataQuery()
    print("Available datasets:")
    for category, files in query.list_available_datasets().items():
        print(f"  {category}: {len(files)} files")
    
    try:
        df = query.get_teacher_inputs()
        print(f"\nTeacher inputs: {len(df):,} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"\n{e}")

