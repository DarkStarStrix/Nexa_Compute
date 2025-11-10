"""Evaluation orchestration routines."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.config import load_config  # type: ignore
from nexa_compute.core.artifacts import ArtifactMeta, create_artifact  # type: ignore
from nexa_compute.evaluation import Evaluator  # type: ignore
from nexa_compute.data import DataPipeline  # type: ignore
from nexa_compute.models import DEFAULT_MODEL_REGISTRY  # type: ignore
from nexa_compute.utils.checkpoint import load_checkpoint  # type: ignore


_DEF_EVAL_ARTIFACT_DIR = "artifacts/eval"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def evaluate_checkpoint(config_path: Path, checkpoint: Optional[Path] = None) -> ArtifactMeta:
    cfg = load_config(config_path)
    pipeline = DataPipeline(cfg.data)
    dataloader = pipeline.dataloader("validation", batch_size=cfg.evaluation.batch_size)
    model = DEFAULT_MODEL_REGISTRY.build(cfg.model)
    if checkpoint:
        state = load_checkpoint(checkpoint)
        model.load_state_dict(state["model_state"])
    evaluator = Evaluator(cfg.evaluation)
    metrics = evaluator.evaluate(model, dataloader)
    run_dir = cfg.output_directory()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "eval_report.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[nexa-eval] Metrics saved to {output}")

    artifact_dir = run_dir / _DEF_EVAL_ARTIFACT_DIR

    def _producer(tmp_dir: Path) -> ArtifactMeta:
        tmp_report = tmp_dir / output.name
        shutil.copy2(output, tmp_report)
        summary = {
            "config": str(config_path),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "metrics": metrics,
        }
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report_bytes = tmp_report.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
        hasher = hashlib.sha256()
        hasher.update(report_bytes)
        hasher.update(manifest_bytes)
        return ArtifactMeta(
            kind="eval_report",
            uri=str(artifact_dir.resolve()),
            hash=f"sha256:{hasher.hexdigest()}",
            bytes=len(report_bytes) + len(manifest_bytes),
            created_at=_now_utc(),
            inputs=[str(checkpoint)] if checkpoint else [str(config_path)],
            labels={"source": "nexa_eval.analyze"},
        )

    return create_artifact(artifact_dir, _producer)
