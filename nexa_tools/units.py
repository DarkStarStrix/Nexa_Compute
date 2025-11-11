"""Unit conversion utilities backed by the Pint library."""

from __future__ import annotations

from typing import Any, Dict

try:
    from pint import UnitRegistry
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "pint is required for unit conversions. Install it with `pip install pint`."
    ) from exc


class UnitConverter:
    """Thin wrapper around Pint's UnitRegistry."""

    def __init__(self) -> None:
        self._ureg = UnitRegistry()

    def convert(self, value: float | int, *, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert a quantity between units."""

        quantity = self._ureg.Quantity(value, from_unit)
        converted = quantity.to(to_unit)
        return {"value": converted.magnitude, "unit": str(converted.units)}


__all__ = ["UnitConverter"]

