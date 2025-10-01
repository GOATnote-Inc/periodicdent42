from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from .adapters.sim import SimDevice
from .models.schemas import ExperimentPlan, InstrumentType, MeasurementResult, RunRecord, RunStatus
from .state import OrchestratorState
from .utils.run_ids import deterministic_run_id


def run(plan_path: Path, max_steps: int = 10) -> None:
    plan_data = yaml.safe_load(plan_path.read_text())
    plan = ExperimentPlan.parse_obj(plan_data)
    run_id = deterministic_run_id(plan.dict())
    state = OrchestratorState(Path("labloop_data"))
    instrument = SimDevice(plan.instrument, seed=42)
    now = datetime.utcnow()
    record = RunRecord(
        run_id=run_id,
        plan=plan,
        status=RunStatus.QUEUED,
        created_at=now,
        updated_at=now,
        backend="sim",
        resource="SIM",
        plan_path=str(Path("labloop_data") / "data" / "plans" / f"{run_id}.json"),
        events_path="",
        results_path="",
        logs_path="",
    )
    run_state = state.create_run(record, instrument)
    total = 0
    while run_state.pending and total < max_steps:
        decision = run_state.scheduler.propose_next(
            [(task, instrument.estimate_duration(task)) for task in run_state.pending]
        )
        if decision is None:
            break
        task = next(t for t in run_state.pending if t.id == decision.task_id)
        run_state.pending.remove(task)
        measurement = instrument.execute(task, run_state.cancel_token)
        result = MeasurementResult(
            task_id=task.id,
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            data_path=f"memory:{task.id}",
            summary={"points": len(measurement.data.get("points", []))},
        )
        run_state.completed.append(result)
        run_state.scheduler.update_measurement(result)
        record.eig_history.append(decision.eig_per_hour)
        record.predictive_variance.append(decision.candidate_scores[0]["posterior_variance"])
        total += 1
        print(f"Step {total}: task {task.id} EIG/hr {decision.eig_per_hour:.3f}")
    print(json.dumps({
        "run_id": run_id,
        "steps": total,
        "eig_history": record.eig_history,
        "predictive_variance": record.predictive_variance,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    run(args.plan, max_steps=args.max_steps)
