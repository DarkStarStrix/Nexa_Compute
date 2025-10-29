"""Logging utilities with basic structured logging support."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Optional

from .time import utc_timestamp


@dataclass
class LoggingSettings:
    level: str = "INFO"
    log_dir: Optional[str] = None
    json_logs: bool = True
    tensorboard: bool = True


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": utc_timestamp(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_context"):
            payload.update(getattr(record, "extra_context"))
        return json.dumps(payload)


def configure_logging(level: str = "INFO", log_dir: Optional[str] = None, *, json_logs: bool = True) -> None:
    logging.captureWarnings(True)
    logging.root.handlers = []
    logging.root.setLevel(level.upper())
    handler: logging.Handler
    if json_logs:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
    logging.root.addHandler(handler)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "train.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logging.root.addHandler(file_handler)


def get_logger(name: str) -> Logger:
    logger = logging.getLogger(name)
    return logger


def log_with_context(logger: Logger, level: str, message: str, *, extra: Optional[dict[str, Any]] = None) -> None:
    extra_context = {"extra_context": extra or {}}
    logger.log(getattr(logging, level.upper()), message, extra=extra_context)


def in_ci() -> bool:
    return os.environ.get("CI", "") == "true"
