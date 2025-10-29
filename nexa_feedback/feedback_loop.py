"""Feedback loop orchestration."""

from __future__ import annotations

from pathlib import Path

from .weakness_analysis import identify_weaknesses
from .feedback_generators.synthetic import generate_feedback_dataset


def run_feedback_cycle(config_path: Path) -> Path:
    weaknesses = identify_weaknesses(config_path)
    feedback_dataset = generate_feedback_dataset(weaknesses)
    output_dir = Path("runs/manifests")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "feedback_dataset.json"
    target.write_text(feedback_dataset, encoding="utf-8")
    print(f"[nexa-feedback] Feedback dataset materialised at {target}")
    return target
