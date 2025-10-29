"""Generate synthetic feedback payloads."""

from __future__ import annotations

import json
from typing import Dict


def generate_feedback_dataset(weaknesses: Dict[str, float]) -> str:
    payload = {
        "weaknesses": weaknesses,
        "recommendations": [
            "Collect more samples in underperforming classes",
            "Tune learning rate schedule",
        ],
    }
    return json.dumps(payload, indent=2)
