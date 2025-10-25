from __future__ import annotations

import json
import math
import random
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from ..models.schemas import InstrumentType, Task, TaskType
from .base import Instrument, Measurement, PreflightReport


class SimDevice(Instrument):
    """High fidelity simulator for the lab instruments."""

    def __init__(
        self,
        instrument: InstrumentType,
        seed: int | None = None,
        latency_range: Tuple[float, float] = (0.1, 0.3),
        noise_level: float = 0.05,
        output_dir: Path | None = None,
    ) -> None:
        self.instrument = instrument
        self.random = random.Random(seed)
        self.latency_range = latency_range
        self.noise_level = noise_level
        self.output_dir = output_dir
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def preflight(self, plan) -> PreflightReport:  # type: ignore[override]
        if not plan.tasks:
            return PreflightReport(False, "Plan contains no tasks")
        return PreflightReport(True, "Simulated device ready")

    def execute(self, task: Task, cancel_token: threading.Event) -> Measurement:  # type: ignore[override]
        if task.type == TaskType.XRD:
            data = self._simulate_xrd(task.parameters, cancel_token)
        else:
            data = self._simulate_transport(task.parameters, cancel_token)
        if self.output_dir:
            path = self.output_dir / f"{task.id}.json"
            path.write_text(json.dumps(data))
            data = {**data, "artifact": str(path)}
        return Measurement(task.id, data)

    def estimate_duration(self, task: Task) -> timedelta:  # type: ignore[override]
        if task.type == TaskType.XRD:
            steps = self._xrd_angles(task.parameters)
            dwell = float(task.parameters.get("dwell_time", 1.0))
            runtime = steps * dwell
        else:
            num_points = int(task.parameters.get("num_points", 10))
            hold = float(task.parameters.get("hold_time", 0.5))
            runtime = num_points * hold
        runtime += sum(self.latency_range) / 2
        return timedelta(seconds=runtime)

    # Internal helpers -------------------------------------------------
    def _latency(self) -> float:
        low, high = self.latency_range
        return self.random.uniform(low, high)

    def _simulate_xrd(self, params: Dict[str, float], cancel_token: threading.Event) -> Dict[str, Iterable[float]]:
        start = float(params["two_theta_start"])
        end = float(params["two_theta_end"])
        step = float(params["step"])
        dwell = float(params["dwell_time"])
        two_theta = np.arange(start, end + step / 2, step)
        center_peaks = [30, 55, 72]
        intensities = np.zeros_like(two_theta)
        for center in center_peaks:
            intensities += np.exp(-0.5 * ((two_theta - center) / 1.2) ** 2)
        intensities *= 1000
        noise = self.random.gauss(0, self.noise_level)
        intensities *= 1 + noise
        measurements = []
        for angle, intensity in zip(two_theta, intensities):
            if cancel_token.is_set():
                break
            time.sleep(dwell + self._latency())
            jitter = self.random.gauss(0, self.noise_level * 10)
            measurements.append({
                "two_theta": float(angle),
                "intensity": float(max(intensity + jitter, 0)),
            })
        return {"type": "xrd", "points": measurements, "dwell": dwell}

    def _simulate_transport(self, params: Dict[str, float], cancel_token: threading.Event) -> Dict[str, Iterable[float]]:
        start = float(params["current_start"])
        end = float(params["current_end"])
        points = int(params["num_points"])
        hold = float(params["hold_time"])
        temperature = float(params.get("temperature", 300))
        currents = np.linspace(start, end, points)
        base_resistance = 10 + 0.05 * (temperature - 300)
        records = []
        for current in currents:
            if cancel_token.is_set():
                break
            time.sleep(hold + self._latency())
            resistance = base_resistance * (1 + self.random.gauss(0, self.noise_level))
            voltage = current * resistance
            records.append({
                "current": float(current),
                "voltage": float(voltage),
                "resistance": float(resistance),
                "temperature": temperature,
            })
        return {
            "type": "transport",
            "points": records,
        }

    def _xrd_angles(self, params: Dict[str, float]) -> int:
        start = float(params["two_theta_start"])
        end = float(params["two_theta_end"])
        step = float(params["step"])
        return max(1, int(math.ceil((end - start) / step)))
