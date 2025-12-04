"""Logging configuration with structured JSON support and correlation IDs."""

from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Context variable for correlation ID
_CORRELATION_ID: ContextVar[str] = ContextVar("correlation_id", default="")


class JsonFormatter(logging.Formatter):
    """Format logs as structured JSON with trace context."""

    def format(self, record: logging.LogRecord) -> str:
        trace_id = ""
        span_id = ""
        
        # Attempt to get OpenTelemetry trace context if available
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span != trace.NonRecordingSpan(None):
                ctx = span.get_span_context()
                trace_id = f"{ctx.trace_id:032x}"
                span_id = f"{ctx.span_id:016x}"
        except ImportError:
            pass

        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno,
            "correlation_id": _CORRELATION_ID.get(),
            "trace_id": trace_id,
            "span_id": span_id,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_context"):
            payload.update(record.extra_context)  # type: ignore

        return json.dumps(payload)


def get_correlation_id() -> str:
    """Get the current correlation ID or generate a new one."""
    cid = _CORRELATION_ID.get()
    if not cid:
        cid = uuid.uuid4().hex
        _CORRELATION_ID.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    _CORRELATION_ID.set(cid)


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    json_logs: bool = False,
) -> None:
    """Configure global logging."""
    handlers: list[Any] = [logging.StreamHandler(sys.stdout)]
    
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "nexa.log"))

    if json_logs:
        formatter = JsonFormatter()
        for handler in handlers:
            handler.setFormatter(formatter)
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        for handler in handlers:
            handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(
        level=level.upper(),
        handlers=handlers,
        force=True,
    )
    
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)

