"""Feedback collection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List


@dataclass
class FeedbackRecord:
    query_id: str
    result_id: str
    is_correct: bool
    user_id: str | None
    timestamp: str


class FeedbackCollector:
    """In-memory feedback collection. Replace with persistent storage for production."""

    def __init__(self) -> None:
        self.records: List[FeedbackRecord] = []

    def record(self, query_id: str, result_id: str, is_correct: bool, user_id: str | None = None) -> FeedbackRecord:
        record = FeedbackRecord(
            query_id=query_id,
            result_id=result_id,
            is_correct=is_correct,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.records.append(record)
        return record

    def stats(self) -> Dict[str, int]:
        positives = sum(record.is_correct for record in self.records)
        return {
            "total": len(self.records),
            "positives": positives,
            "negatives": len(self.records) - positives,
        }

