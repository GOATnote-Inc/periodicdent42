from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Iterable, List, Tuple

from ..models.schemas import MeasurementResult, SchedulerDecision, Task


@dataclass
class TaskStats:
    completed: int = 0
    variance: float = 1.0
    duration_s: float = 30.0


@dataclass
class SchedulerState:
    task_stats: Dict[str, TaskStats] = field(default_factory=dict)


class EIGScheduler:
    """Lightweight scheduler that ranks tasks by EIG/hour.

    The entropy proxy assumes a Gaussian posterior whose variance shrinks as
    more observations are collected. While simplistic, it provides a monotonic
    decrease in predicted variance which is sufficient for the vertical slice.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = SchedulerState()

    def update_measurement(self, result: MeasurementResult) -> None:
        with self._lock:
            stats = self._state.task_stats.setdefault(result.task_id, TaskStats())
            stats.completed += 1
            stats.variance = max(0.05, stats.variance * 0.7)
            duration = result.summary.get("duration_s")
            if duration:
                stats.duration_s = 0.5 * stats.duration_s + 0.5 * float(duration)

    def rank_tasks(self, tasks: Iterable[Tuple[Task, timedelta]]) -> List[Dict[str, float]]:
        rankings: List[Dict[str, float]] = []
        with self._lock:
            for task, duration in tasks:
                stats = self._state.task_stats.setdefault(task.id, TaskStats())
                sigma_prior = stats.variance
                sigma_post = max(0.05, sigma_prior * 0.7)
                eig_bits = 0.5 * math.log((sigma_prior ** 2) / (sigma_post ** 2), 2)
                hours = max(duration.total_seconds() / 3600.0, 1e-3)
                eig_per_hour = eig_bits / hours
                rankings.append(
                    {
                        "task_id": task.id,
                        "eig_bits": eig_bits,
                        "duration_hours": hours,
                        "eig_per_hour": eig_per_hour,
                        "prior_variance": sigma_prior,
                        "posterior_variance": sigma_post,
                    }
                )
                stats.duration_s = duration.total_seconds()
        rankings.sort(key=lambda item: item["eig_per_hour"], reverse=True)
        return rankings

    def propose_next(self, tasks: Iterable[Tuple[Task, timedelta]]) -> SchedulerDecision | None:
        ranked = self.rank_tasks(tasks)
        if not ranked:
            return None
        top = ranked[0]
        rationale = (
            f"Selected task {top['task_id']} with EIG {top['eig_bits']:.3f} bits "
            f"and estimated duration {top['duration_hours']:.3f} h"
        )
        return SchedulerDecision(
            task_id=top["task_id"],
            eig_bits=top["eig_bits"],
            duration_hours=top["duration_hours"],
            eig_per_hour=top["eig_per_hour"],
            rationale=rationale,
            candidate_scores=ranked,
        )
