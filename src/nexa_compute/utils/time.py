"""Time helpers."""

from __future__ import annotations

import datetime as dt


def utc_timestamp() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
