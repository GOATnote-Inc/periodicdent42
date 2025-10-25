from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np

from ..models.schemas import WorkflowEvent


@dataclass
class StepTiming:
    step: str
    enter: datetime
    exit: datetime

    @property
    def duration(self) -> float:
        return (self.exit - self.enter).total_seconds()


class ValueStreamMapper:
    def __init__(self, steps: List[str]):
        self.steps = steps

    def map(self, events: Iterable[WorkflowEvent]) -> Dict[str, List[StepTiming]]:
        per_unit: Dict[str, Dict[str, Dict[str, datetime]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for event in events:
            per_unit[event.unit_id][event.step][event.state] = event.ts
        timings: Dict[str, List[StepTiming]] = defaultdict(list)
        for unit_id, steps in per_unit.items():
            for step in self.steps:
                state = steps.get(step, {})
                if "entered" in state and "exited" in state:
                    timings[unit_id].append(
                        StepTiming(step=step, enter=state["entered"], exit=state["exited"])
                    )
        return timings

    def compute_cycle_metrics(self, timings: Dict[str, List[StepTiming]]):
        metrics = []
        for unit_id, unit_timings in timings.items():
            if not unit_timings:
                continue
            total_cycle = unit_timings[-1].exit - unit_timings[0].enter
            touch = sum(t.duration for t in unit_timings)
            wait = total_cycle.total_seconds() - touch
            metrics.append(
                {
                    "unit_id": unit_id,
                    "cycle_time_s": total_cycle.total_seconds(),
                    "touch_time_s": touch,
                    "wait_time_s": max(wait, 0.0),
                    "start_ts": unit_timings[0].enter,
                    "end_ts": unit_timings[-1].exit,
                }
            )
        return metrics


def summarize_metrics(records: List[dict]) -> dict:
    if not records:
        return {
            "cycle_time_p50": 0.0,
            "cycle_time_p90": 0.0,
            "throughput_per_day": 0.0,
            "wip": 0.0,
            "yield_rate": 0.0,
            "defects": {},
            "units": 0,
        }
    cycle_times = np.array([r["cycle_time_s"] for r in records])
    p50 = float(np.percentile(cycle_times, 50))
    p90 = float(np.percentile(cycle_times, 90))
    start = min(r["start_ts"] for r in records)
    end = max(r["end_ts"] for r in records)
    days = max((end - start).total_seconds() / 86400.0, 1.0)
    throughput = len(records) / days
    wip = throughput * p50 / 86400.0
    yield_rate = np.mean([1 if r.get("yield_ok", True) else 0 for r in records])
    defects = defaultdict(int)
    for record in records:
        code = record.get("defect_code") or "ok"
        defects[code] += 1
    return {
        "cycle_time_p50": p50,
        "cycle_time_p90": p90,
        "throughput_per_day": throughput,
        "wip": wip,
        "yield_rate": float(yield_rate),
        "defects": dict(defects),
        "units": len(records),
    }
