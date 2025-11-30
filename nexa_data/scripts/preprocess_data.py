"""Preprocess raw datasets into train/validation/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def preprocess(
    raw_path: Path = typer.Option(Path("data/raw/synthetic.csv"), help="Raw dataset path"),
    output_dir: Path = typer.Option(Path("data/processed"), help="Directory for processed data"),
    validation_fraction: float = typer.Option(0.1, min=0.0, max=0.5),
    test_fraction: float = typer.Option(0.1, min=0.0, max=0.5),
    seed: int = typer.Option(42),
) -> None:
    """Shuffle and split the raw dataset into train/validation/test parquet files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        typer.echo("Raw dataset missing, generating synthetic placeholder...")
        _generate_synthetic(raw_path, seed=seed)
    df = pd.read_csv(raw_path)
    df = df.sample(frac=1.0, random_state=seed)
    n_total = len(df)
    n_val = int(n_total * validation_fraction)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_val - n_test

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    train_df.to_parquet(output_dir / "train.parquet")
    val_df.to_parquet(output_dir / "val.parquet")
    test_df.to_parquet(output_dir / "test.parquet")

    metadata = {
        "rows": n_total,
        "columns": list(df.columns),
        "splits": {"train": n_train, "validation": n_val, "test": n_test},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    typer.echo(f"Processed dataset saved to {output_dir}")


def _generate_synthetic(path: Path, *, seed: int) -> None:
    """Create a synthetic dataset for local development."""

    rng = np.random.default_rng(seed)
    num_samples = 5000
    features = rng.normal(size=(num_samples, 4))
    weights = np.array([0.4, -0.2, 0.1, 0.5])
    logits = features @ weights + rng.normal(scale=0.1, size=(num_samples,))
    labels = (logits > logits.mean()).astype(int)
    df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
    df["label"] = labels
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    app()
