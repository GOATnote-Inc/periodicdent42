from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import random
from statistics import pstdev


@dataclass
class SurrogateObservation:
    x: Sequence[float]
    y: float
    duration_min: float


@dataclass
class GPSurrogate:
    observations: List[SurrogateObservation] = field(default_factory=list)

    def update(self, x: Sequence[float], y: float, duration_min: float) -> None:
        self.observations.append(SurrogateObservation(x=list(x), y=float(y), duration_min=float(duration_min)))

    def propose_candidates(self, k: int = 3, bounds: Sequence[Tuple[float, float]] | None = None) -> List[Tuple[List[float], float]]:
        if bounds is None:
            dimension = len(self.observations[0].x) if self.observations else 1
            bounds = [(0.0, 1.0)] * dimension
        candidates: List[Tuple[List[float], float]] = []
        noise = self._observed_std() or 1.0
        for _ in range(k):
            point = [random.uniform(low, high) for low, high in bounds]
            candidates.append((point, noise))
        return candidates

    def eig_per_hour(self, candidate_std: float, expected_duration_min: float) -> float:
        if expected_duration_min <= 0:
            return 0.0
        duration_hours = expected_duration_min / 60.0
        return (candidate_std + 1.0) / max(duration_hours, 1e-6)

    def _observed_std(self) -> float:
        if not self.observations:
            return 0.0
        values = [obs.y for obs in self.observations]
        return pstdev(values)
