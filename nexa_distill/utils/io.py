"""I/O utilities for working with parquet and JSONL artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for ``path`` if it is missing."""

    path.parent.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""

    return pd.read_parquet(path, columns=list(columns) if columns else None)


def write_parquet(df: pd.DataFrame, path: Path, *, compression: str = "zstd") -> None:
    """Write a DataFrame to parquet, ensuring the destination exists."""

    ensure_parent_dir(path)
    df.to_parquet(path, index=False, compression=compression)


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield dictionaries from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    """Write an iterable of dictionaries to JSON Lines format."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def list_to_jsonl(records: List[dict], path: Path) -> None:
    """Compatibility helper to mirror JSONL writing from a list."""

    write_jsonl(records, path)

