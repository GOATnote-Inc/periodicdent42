from datetime import timedelta

from datetime import datetime

from labloop.orchestrator.models.schemas import MeasurementResult, Task, TaskType
from labloop.orchestrator.scheduler.eig_scheduler import EIGScheduler


def test_scheduler_monotonic_variance():
    scheduler = EIGScheduler()
    task = Task(id="t1", type=TaskType.XRD, parameters={})
    ranked = scheduler.rank_tasks([(task, timedelta(seconds=10))])
    prior = ranked[0]["prior_variance"]
    decision = scheduler.propose_next([(task, timedelta(seconds=10))])
    assert decision is not None
    now = datetime.utcnow()
    result = MeasurementResult(
        task_id="t1",
        started_at=now,
        finished_at=now,
        data_path="/tmp",
        summary={"duration_s": 10},
    )
    scheduler.update_measurement(result)
    ranked_after = scheduler.rank_tasks([(task, timedelta(seconds=10))])
    assert ranked_after[0]["prior_variance"] < prior
