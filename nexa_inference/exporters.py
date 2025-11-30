"""Export utilities for embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def export_mgf(embeddings: np.ndarray, spectrum_ids: List[str], output_path: Path) -> Path:
    """Export embeddings to simplified MGF file."""
    lines = []
    for spectrum_id, vector in zip(spectrum_ids, embeddings):
        lines.append("BEGIN IONS")
        lines.append(f"TITLE={spectrum_id}")
        lines.append("PEPMASS=0")
        for idx, value in enumerate(vector):
            lines.append(f"{idx} {value}")
        lines.append("END IONS")
    output_path.write_text("\n".join(lines))
    return output_path


def export_json_schema(embeddings: np.ndarray, spectrum_ids: List[str], output_path: Path) -> Path:
    """Export embeddings to JSON schema friendly format."""
    payload = [
        {"spectrum_id": spectrum_id, "embedding": vector.tolist()} for spectrum_id, vector in zip(spectrum_ids, embeddings)
    ]
    import json

    output_path.write_text(json.dumps(payload, indent=2))
    return output_path

