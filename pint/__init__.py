"""Lightweight stub of pint.UnitRegistry for offline testing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Quantity:
    magnitude: float
    unit: str

    def to_base_units(self) -> "_Quantity":
        return self


class UnitRegistry:
    """Minimal pint.UnitRegistry interface for unit validation."""

    def __call__(self, unit: str) -> str:
        if not isinstance(unit, str) or not unit:
            raise ValueError("Unit must be a non-empty string")
        return unit

    def Quantity(self, value: float, unit: str) -> _Quantity:  # noqa: N802 - mimic pint API
        self(unit)
        return _Quantity(magnitude=value, unit=unit)
