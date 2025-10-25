"""Base classes and contracts for physical instrument drivers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


class SafetyInterlockError(RuntimeError):
    """Raised when attempting to operate an instrument in an unsafe state."""


@dataclass
class InstrumentConfiguration:
    """Generic configuration payload for an instrument."""

    integration_time_ms: int = 100
    averages: int = 3
    dark_reference_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstrumentStatus:
    """Run-time status for monitoring and telemetry."""

    instrument_id: str
    connected: bool
    state: str
    last_heartbeat: datetime
    temperature_c: float
    faults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementPayload:
    """Standardized payload returned by a driver measurement."""

    experiment_id: str
    instrument_id: str
    wavelengths_nm: list[float]
    absorbance: list[float]
    metadata: Dict[str, Any]
    captured_at: datetime


class InstrumentDriver(ABC):
    """Interface that all instrument drivers must implement."""

    instrument_id: str

    def __init__(self, instrument_id: str) -> None:
        self.instrument_id = instrument_id
        self._connected = False
        self._state = "disconnected"
        self._last_status = InstrumentStatus(
            instrument_id=instrument_id,
            connected=False,
            state="disconnected",
            last_heartbeat=datetime.utcnow(),
            temperature_c=23.5,
        )

    @property
    def status(self) -> InstrumentStatus:
        """Return latest instrument status."""

        return self._last_status

    def _update_status(self, **kwargs: Any) -> None:
        """Internal helper to update cached status."""

        data = self._last_status.__dict__.copy()
        data.update(kwargs)
        data.setdefault("last_heartbeat", datetime.utcnow())
        self._last_status = InstrumentStatus(**data)

    @abstractmethod
    def connect(self) -> None:
        """Establish connection with the instrument."""

    @abstractmethod
    def disconnect(self) -> None:
        """Safely disconnect from the instrument."""

    @abstractmethod
    def initialize(self, configuration: InstrumentConfiguration) -> None:
        """Apply configuration and perform warm-up routines."""

    @abstractmethod
    def perform_measurement(self, *, experiment_id: str, sample_metadata: Dict[str, Any]) -> MeasurementPayload:
        """Execute a measurement sequence and return structured data."""

    @abstractmethod
    def safety_check(self) -> None:
        """Validate that instrument is in a safe state for operation."""

    def ensure_connected(self) -> None:
        """Ensure the instrument is connected before running commands."""

        if not self._connected:
            raise SafetyInterlockError("Instrument not connected")

    def ensure_state(self, expected: str) -> None:
        """Ensure instrument is in the expected state."""

        if self._state != expected:
            raise SafetyInterlockError(f"Instrument state {self._state} != expected {expected}")
