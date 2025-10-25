"""Simulated UV-Vis spectrometer driver with safety interlocks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.drivers.base import (
    InstrumentConfiguration,
    InstrumentDriver,
    MeasurementPayload,
    SafetyInterlockError,
)


@dataclass
class UvVisConfiguration(InstrumentConfiguration):
    """Configuration knobs specific to the UV-Vis spectrometer."""

    wavelength_start_nm: float = 350.0
    wavelength_end_nm: float = 800.0
    wavelength_step_nm: float = 5.0
    lamp_warmup_minutes: int = 10
    allow_auto_dark_reference: bool = True
    max_power_mw: float = 5.0
    facility: Optional[str] = None
    safety_contact: Optional[str] = None


@dataclass
class _OpticalPath:
    """Internal representation of the optical path state."""

    cuvette_inserted: bool = False
    lid_closed: bool = False
    dark_reference: Optional[List[float]] = None
    last_dark_reference: Optional[datetime] = None
    last_sample_id: Optional[str] = None


class UvVisSpectrometer(InstrumentDriver):
    """Simulated UV-Vis spectrometer driver with rich telemetry."""

    def __init__(self, instrument_id: str = "uvvis-001") -> None:
        super().__init__(instrument_id)
        self._configuration = UvVisConfiguration()
        self._optics = _OpticalPath()
        self._lamp_ready_at: Optional[datetime] = None
        self._temperature_c = 23.0
        self._drift_per_hour = 0.05
        self._faults: Dict[str, str] = {}
        self._baseline_noise = 0.0025
        self._rng = random.Random(42)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        self._connected = True
        self._state = "idle"
        self._update_status(
            connected=True,
            state="idle",
            temperature_c=self._temperature_c,
            faults=self._faults,
        )

    def disconnect(self) -> None:
        self.ensure_connected()
        self._connected = False
        self._state = "disconnected"
        self._update_status(connected=False, state="disconnected")

    # ------------------------------------------------------------------
    # Configuration and safety
    # ------------------------------------------------------------------
    def initialize(self, configuration: InstrumentConfiguration) -> None:
        self.ensure_connected()
        if not isinstance(configuration, UvVisConfiguration):
            raise ValueError("UV-Vis driver requires UvVisConfiguration")

        self._configuration = configuration
        warmup_duration = timedelta(minutes=configuration.lamp_warmup_minutes)
        self._lamp_ready_at = datetime.utcnow() + warmup_duration
        self._state = "warming"
        self._update_status(state="warming")

    def complete_warmup(self) -> None:
        """Advance the simulation to mark the lamp as ready."""

        self.ensure_connected()
        if self._lamp_ready_at and datetime.utcnow() >= self._lamp_ready_at:
            self._state = "ready"
            self._update_status(state="ready")
        else:
            raise SafetyInterlockError("Lamp is still warming up")

    def record_dark_reference(self, sample_id: str) -> None:
        """Record a dark reference spectrum for subsequent background subtraction."""

        self.ensure_connected()
        self.ensure_state("ready")
        wavelengths = self._generate_wavelengths()
        baseline = [self._rng.gauss(0.0, self._baseline_noise) for _ in wavelengths]
        self._optics.dark_reference = baseline
        self._optics.last_dark_reference = datetime.utcnow()
        self._optics.last_sample_id = sample_id

    def force_ready(self) -> None:
        """Utility for simulations to mark the lamp as ready immediately."""

        self.ensure_connected()
        self._lamp_ready_at = datetime.utcnow()
        self._state = "ready"
        self._update_status(state="ready")

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------
    def insert_cuvette(self, sample_id: str) -> None:
        self.ensure_connected()
        self._optics.cuvette_inserted = True
        self._optics.last_sample_id = sample_id

    def eject_cuvette(self) -> None:
        self.ensure_connected()
        self._optics.cuvette_inserted = False
        self._optics.last_sample_id = None

    def close_lid(self) -> None:
        self.ensure_connected()
        self._optics.lid_closed = True

    def open_lid(self) -> None:
        self.ensure_connected()
        self._optics.lid_closed = False

    def safety_check(self) -> None:
        self.ensure_connected()
        faults: Dict[str, str] = {}
        if self._state not in {"warming", "ready", "measuring", "idle"}:
            faults["state"] = f"Unexpected state: {self._state}"
        if self._state == "ready" and not self._lamp_ready_at:
            faults["lamp"] = "Lamp warm-up not scheduled"
        if self._state == "measuring" and not self._optics.lid_closed:
            faults["lid"] = "Measurement attempted with lid open"
        if self._optics.cuvette_inserted and not self._optics.lid_closed:
            faults["lid"] = "Lid must be closed when cuvette inserted"
        self._faults = faults
        if faults:
            self._update_status(faults=faults)
            raise SafetyInterlockError("Safety check failed: " + ", ".join(faults.values()))

    # ------------------------------------------------------------------
    # Measurement logic
    # ------------------------------------------------------------------
    def perform_measurement(self, *, experiment_id: str, sample_metadata: Dict[str, Any]) -> MeasurementPayload:
        self.ensure_connected()
        if self._state == "warming":
            self.complete_warmup()
        self.ensure_state("ready")
        if not self._optics.cuvette_inserted:
            raise SafetyInterlockError("No cuvette inserted")
        if not self._optics.lid_closed:
            raise SafetyInterlockError("Lid must be closed for measurement")
        if self._lamp_ready_at and datetime.utcnow() < self._lamp_ready_at:
            raise SafetyInterlockError("Lamp warm-up incomplete")

        if self._configuration.allow_auto_dark_reference:
            if not self._optics.dark_reference or self._dark_reference_expired():
                self.record_dark_reference(sample_metadata.get("sample_id", "unknown"))

        self._state = "measuring"
        self._update_status(state="measuring")

        wavelengths = self._generate_wavelengths()
        absorbance = self._simulate_absorbance(wavelengths, sample_metadata)

        if self._configuration.allow_auto_dark_reference:
            absorbance = [a - d for a, d in zip(absorbance, self._optics.dark_reference)]

        payload = MeasurementPayload(
            experiment_id=experiment_id,
            instrument_id=self.instrument_id,
            wavelengths_nm=wavelengths,
            absorbance=absorbance,
            metadata={
                "integration_time_ms": self._configuration.integration_time_ms,
                "averages": self._configuration.averages,
                "facility": self._configuration.facility,
                "safety_contact": self._configuration.safety_contact,
                "sample_id": sample_metadata.get("sample_id"),
                "composition": sample_metadata.get("composition"),
                "labels": sample_metadata.get("labels"),
                "temperature_c": round(self._temperature_c, 2),
            },
            captured_at=datetime.utcnow(),
        )

        self._state = "ready"
        self._update_status(state="ready", temperature_c=self._temperature_c)
        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_wavelengths(self) -> List[float]:
        cfg = self._configuration
        count = int((cfg.wavelength_end_nm - cfg.wavelength_start_nm) / cfg.wavelength_step_nm) + 1
        return [cfg.wavelength_start_nm + i * cfg.wavelength_step_nm for i in range(count)]

    def _simulate_absorbance(self, wavelengths: List[float], sample_metadata: Dict[str, Any]) -> List[float]:
        peak_nm = float(sample_metadata.get("labels", {}).get("peak_nm", 520.0))
        width = float(sample_metadata.get("metadata", {}).get("peak_width_nm", 40.0))
        amplitude = float(sample_metadata.get("metadata", {}).get("peak_absorbance", 1.0))
        baseline = float(sample_metadata.get("metadata", {}).get("baseline", 0.05))

        spectrum: List[float] = []
        for wl in wavelengths:
            gaussian = amplitude * math.exp(-0.5 * ((wl - peak_nm) / width) ** 2)
            noise = self._rng.gauss(0.0, self._baseline_noise)
            drift = self._compute_drift()
            spectrum.append(baseline + gaussian + noise + drift)
        return spectrum

    def _compute_drift(self) -> float:
        elapsed_hours = self._rng.random() * 0.25
        return self._drift_per_hour * elapsed_hours

    def _dark_reference_expired(self) -> bool:
        if not self._optics.last_dark_reference:
            return True
        return (datetime.utcnow() - self._optics.last_dark_reference) > timedelta(minutes=30)
