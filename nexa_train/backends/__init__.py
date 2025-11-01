"""Backend registry for Nexa Train launchers."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict


BackendFn = Callable[[list[str] | None], None]

_BACKEND_CACHE: Dict[str, BackendFn] = {}


def get_backend(name: str) -> BackendFn:
    """Return a backend entrypoint by name."""

    normalized = name.lower()
    if normalized not in _BACKEND_CACHE:
        module_name, func_name = _BACKEND_SPECS[normalized]
        module = import_module(module_name)
        _BACKEND_CACHE[normalized] = getattr(module, func_name)
    return _BACKEND_CACHE[normalized]


_BACKEND_SPECS: Dict[str, tuple[str, str]] = {
    "hf": ("nexa_train.backends.hf_runner", "main"),
}


__all__ = ["get_backend", "BackendFn"]

