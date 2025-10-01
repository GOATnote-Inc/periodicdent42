from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List

from ..models.schemas import WorkflowEvent
from .value_stream import ValueStreamMapper


class MetricEmitter:
    def __init__(self, steps: List[str]):
        self.mapper = ValueStreamMapper(steps)

    def emit(self, events: Iterable[WorkflowEvent]) -> List[dict]:
        events_list = list(events)
        timings = self.mapper.map(events_list)
        records = self.mapper.compute_cycle_metrics(timings)
        defect_map: Dict[str, str | None] = defaultdict(lambda: None)
        for event in events_list:
            if event.step == "done" and event.state == "exited":
                defect_map[event.unit_id] = event.attrs.get("defect_code")
        for record in records:
            defect = defect_map[record["unit_id"]]
            record["defect_code"] = defect
            record["yield_ok"] = defect is None
            record["period"] = record["end_ts"].date().isoformat()
        return records


def split_baseline_pilot(records: List[dict], split_ts: datetime) -> tuple[List[dict], List[dict]]:
    baseline, pilot = [], []
    for record in records:
        if record["end_ts"] <= split_ts:
            baseline.append(record)
        else:
            pilot.append(record)
    return baseline, pilot
