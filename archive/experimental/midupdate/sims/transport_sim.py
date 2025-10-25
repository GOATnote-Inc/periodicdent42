from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


class TransportSimulator:
    def __init__(self, noise: float = 0.5) -> None:
        self.noise = noise

    def run(self, params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        temp = params.get("anneal_temp_C", 450)
        time_min = params.get("anneal_time_min", 30)
        doping = params.get("doping_pct", 0.05)
        tc = 90 - 0.0008 * (temp - 750) ** 2 + 5 * math.exp(-((doping - 0.07) ** 2) / 0.002)
        tc += -0.05 * abs(time_min - 45)
        tc += np.random.normal(0, self.noise)
        duration = 30 + 0.05 * abs(temp - 600)
        qc = tc > 70
        return tc, {"critical_temperature_K": tc, "duration_min": duration, "qc_passed": qc}
