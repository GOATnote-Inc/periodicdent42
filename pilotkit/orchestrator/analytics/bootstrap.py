from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np


def bootstrap_ci(
    baseline: Iterable[float],
    pilot: Iterable[float],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    iterations: int = 2000,
    ci: Tuple[float, float] = (2.5, 97.5),
    seed: int | None = 13,
) -> tuple[float, tuple[float, float]]:
    base = np.array(list(baseline))
    pilot_arr = np.array(list(pilot))
    if base.size == 0 or pilot_arr.size == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(iterations):
        base_sample = rng.choice(base, size=base.size, replace=True)
        pilot_sample = rng.choice(pilot_arr, size=pilot_arr.size, replace=True)
        boot.append(statistic(base_sample, pilot_sample))
    point = statistic(base, pilot_arr)
    lower, upper = np.percentile(boot, list(ci))
    return point, (float(lower), float(upper))
