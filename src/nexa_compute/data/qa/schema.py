"""Lightweight schema validation utilities for dataset quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import json

__all__ = [
    "SchemaCheckResult",
    "validate_records",
    "validate_jsonl_file",
]


@dataclass(frozen=True)
class SchemaCheckResult:
    """Outcome of a schema validation."""

    passed: bool
    messages: Sequence[str] = field(default_factory=tuple)

    def report(self) -> str:
        return "\n".join(self.messages)


def _require_fields(record: Mapping[str, object], required_fields: Sequence[str]) -> list[str]:
    missing = [field for field in required_fields if field not in record]
    return missing


def validate_records(
    records: Iterable[Mapping[str, object]],
    *,
    required_fields: Sequence[str],
    minimum_rows: int | None = None,
) -> SchemaCheckResult:
    """Validate that a collection of records satisfies basic schema properties."""

    messages: list[str] = []
    row_count = 0
    for row_count, record in enumerate(records, start=1):
        missing = _require_fields(record, required_fields)
        if missing:
            messages.append(f"row {row_count}: missing fields {missing}")
    if minimum_rows is not None and row_count < minimum_rows:
        messages.append(f"expected at least {minimum_rows} rows but found {row_count}")
    return SchemaCheckResult(passed=not messages, messages=messages)


def validate_jsonl_file(
    path: Path,
    *,
    required_fields: Sequence[str],
    minimum_rows: int | None = None,
) -> SchemaCheckResult:
    """Validate a JSONL document against simple schema constraints."""

    records: list[Mapping[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                return SchemaCheckResult(passed=False, messages=[f"line {line_number}: invalid JSON ({exc})"])
            if not isinstance(payload, Mapping):
                return SchemaCheckResult(passed=False, messages=[f"line {line_number}: expected object record"])
            records.append(payload)
    return validate_records(records, required_fields=required_fields, minimum_rows=minimum_rows)
