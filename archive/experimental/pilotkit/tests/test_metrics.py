from datetime import datetime, timedelta

from datetime import datetime, timedelta

from pilotkit.orchestrator.analytics.metrics import MetricEmitter, split_baseline_pilot
from pilotkit.orchestrator.analytics.value_stream import summarize_metrics
from pilotkit.orchestrator.models.schemas import WorkflowEvent


def make_events(start: datetime) -> list[WorkflowEvent]:
    events = []
    steps = ["queue", "work", "review", "done"]
    for idx in range(5):
        unit_id = f"unit-{idx}"
        current = start + timedelta(hours=idx)
        for step in steps:
            events.append(
                WorkflowEvent(ts=current, unit_id=unit_id, step=step, state="entered", attrs={})
            )
            current += timedelta(minutes=20)
            events.append(
                WorkflowEvent(ts=current, unit_id=unit_id, step=step, state="exited", attrs={})
            )
        # mark yield defect for one unit
        if idx == 0:
            events[-1].attrs["defect_code"] = "rework"
    return events


def test_cycle_time_and_yield_calculation():
    start = datetime.utcnow() - timedelta(days=10)
    events = make_events(start)
    emitter = MetricEmitter(["queue", "work", "review", "done"])
    records = emitter.emit(events)
    assert records, "metrics should be emitted"
    summary = summarize_metrics(records)
    assert summary["units"] == 5
    assert summary["defects"]["ok"] == 4
    assert summary["defects"]["rework"] == 1


def test_split_baseline_pilot():
    start = datetime.utcnow() - timedelta(days=10)
    events = make_events(start) + make_events(start + timedelta(days=8))
    emitter = MetricEmitter(["queue", "work", "review", "done"])
    records = emitter.emit(events)
    split = datetime.utcnow() - timedelta(days=5)
    baseline, pilot = split_baseline_pilot(records, split)
    assert len(baseline) > 0
    assert len(pilot) > 0
