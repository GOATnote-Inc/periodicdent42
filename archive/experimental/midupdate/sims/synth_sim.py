from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


class SynthesisYieldSimulator:
    def __init__(self, noise: float = 1.0) -> None:
        self.noise = noise

    def run(self, params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        ph = params.get("ph", 7.2)
        stir = params.get("stir_rate_rpm", 500.0)
        feed = params.get("feed_rate_ml_per_min", 5.0)
        yield_pct = 65 + 8 * math.exp(-((ph - 7.8) ** 2) / 0.6)
        yield_pct += 4 * math.exp(-((stir - 520) ** 2) / 40000)
        yield_pct += -1.5 * abs(feed - 4.5)
        yield_pct += np.random.normal(0, self.noise)
        duration = 25 + feed
        qc = yield_pct > 60
        return yield_pct, {"yield_pct": yield_pct, "duration_min": duration, "qc_passed": qc}
