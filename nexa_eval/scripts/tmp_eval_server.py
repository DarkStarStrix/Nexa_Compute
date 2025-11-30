from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project packages are importable
CURRENT_PATH = Path(__file__).resolve()
REPO_ROOT = CURRENT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nexa_eval.rubrics import JUDGE_F_RUBRIC, JUDGE_R_RUBRIC
from safetensors.torch import load_file

CHECKPOINT_ROOT = Path("/Users/allanmurimiwandia/Nexa_compute/nexa_inference/checkpoints/falcon3_qlora_v1/checkpoint-200")
REPORT_PATH = Path("/Users/allanmurimiwandia/Nexa_compute/data/processed/evaluation/reports/distillation_eval_v1.json")

app = FastAPI(title="Nexa Eval Judge", description="FastAPI wrapper around Nexa evaluation rubrics")

adapter_status: dict[str, str] | None = None
adapter_state_keys: list[str] = []

if CHECKPOINT_ROOT.exists():
    try:
        adapter_state = load_file(CHECKPOINT_ROOT / "adapter_model.safetensors")
        adapter_state_keys = sorted(adapter_state.keys())[:5]
        adapter_status = {"status": "loaded"}
    except Exception as exc:  # noqa: BLE001
        adapter_status = {"status": "error", "detail": str(exc)}
else:
    adapter_status = {"status": "missing"}

cached_report: dict[str, object] | None = None
if REPORT_PATH.exists():
    try:
        cached_report = json.loads(REPORT_PATH.read_text())
    except Exception as exc:  # noqa: BLE001
        cached_report = {"error": f"Failed to read report: {exc}"}


class JudgeRequest(BaseModel):
    question: str
    response: str
    context: Optional[str] = None


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "checkpoint": adapter_status,
        "sample_state_keys": adapter_state_keys,
        "rubrics": [JUDGE_F_RUBRIC.name, JUDGE_R_RUBRIC.name],
        "cached_report": bool(cached_report),
    }


@app.get("/report")
def report() -> dict[str, object]:
    if cached_report is None:
        raise HTTPException(status_code=404, detail="No cached evaluation report")
    return cached_report


@app.post("/judge")
def judge(_: JudgeRequest) -> dict[str, object]:
    raise HTTPException(status_code=501, detail="Live judging requires OpenAI access and is not enabled in this stub server")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
