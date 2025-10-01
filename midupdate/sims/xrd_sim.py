from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


class XRDSharpnessSimulator:
    def __init__(self, noise: float = 0.01) -> None:
        self.noise = noise

    def run(self, params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        scan_rate = params.get("scan_rate_deg_per_min", 1.0)
        voltage = params.get("voltage_kV", 40.0)
        height = params.get("sample_height_mm", 0.5)
        peak = 0.8 + 0.1 * math.exp(-((scan_rate - 1.5) ** 2) / 1.5)
        peak += 0.05 * math.exp(-((voltage - 45) ** 2) / 150)
        peak += -0.02 * abs(height - 0.45)
        peak += np.random.normal(0, self.noise)
        duration = 10 + 2.0 / scan_rate
        qc = peak > 0.7
        return peak, {"peak_sharpness": peak, "duration_min": duration, "qc_passed": qc}
