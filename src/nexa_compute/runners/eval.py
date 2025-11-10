"""Evaluation runner emitting eval_report artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from nexa_eval.analyze import evaluate_checkpoint

from ..core.artifacts import ArtifactMeta, create_artifact

__all__ = ["EvalRunSpec", "EvalResult", "EvalRunner"]


@dataclass(frozen=True)
class EvalRunSpec:
    """Configuration describing an evaluation run."""

    config_path: Path
    checkpoint_path: Optional[Path]
    artifact_path: Path


@dataclass(frozen=True)
class EvalResult:
    """Wrapper for evaluation results."""

    artifact: ArtifactMeta


class EvalRunner:
    """Invoke :mod:`nexa_eval` to produce evaluation artifacts."""

    def run(self, spec: EvalRunSpec) -> EvalResult:
        report_path = evaluate_checkpoint(spec.config_path, spec.checkpoint_path)

        def _producer(tmp_dir: Path) -> ArtifactMeta:
            target_report = tmp_dir / report_path.name
            shutil.copy2(report_path, target_report)

            manifest = {
                "config": str(spec.config_path),
                "checkpoint": str(spec.checkpoint_path) if spec.checkpoint_path else None,
                "report": target_report.name,
            }
            manifest_path = tmp_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            payload_hash = hashlib.sha256(target_report.read_bytes()).hexdigest()
            total_bytes = target_report.stat().st_size + manifest_path.stat().st_size

            metadata = ArtifactMeta(
                kind="eval_report",
                uri=str(spec.artifact_path.resolve()),
                hash=f"sha256:{payload_hash}",
                bytes=total_bytes,
                created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                inputs=[str(spec.checkpoint_path)] if spec.checkpoint_path else [],
                labels={"source": "nexa_eval"},
            )
            return metadata

        artifact = create_artifact(spec.artifact_path, _producer)
        return EvalResult(artifact=artifact)
