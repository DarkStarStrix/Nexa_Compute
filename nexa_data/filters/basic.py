"""Placeholder module for dataset filtering logic."""

from __future__ import annotations

from typing import Iterable


def filter_by_label(records: Iterable[tuple], allowed_labels: set[int]):
    for features, label in records:
        if label in allowed_labels:
            yield features, label
