"""Logging helpers for Nexa Distill."""

from __future__ import annotations

import logging
from typing import Optional


_LOGGER_INITIALIZED = False


def _configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger once with a concise formatter."""

    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.propagate = False
    _LOGGER_INITIALIZED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger configured for Nexa Distill."""

    _configure_root_logger()
    return logging.getLogger(name or "nexa_distill")

