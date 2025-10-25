from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .adapters.real import build_real_device
from .adapters.sim import SimDevice
from .models.schemas import (
    AbortRequest,
    ExperimentPlan,
    MeasurementResult,
    ProvenanceGraph,
    RunNextRequest,
    RunRecord,
    RunStatus,
    SchedulerDecision,
)
from .state import OrchestratorState
from .utils.run_ids import deterministic_run_id, feature_flag

LOGGER = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(os.getenv("LABLOOP_DATA", "labloop_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = DATA_DIR / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LabLoop Orchestrator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = OrchestratorState(DATA_DIR)


class SubmitResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunStatusResponse(BaseModel):
    record: RunRecord
    completed: List[MeasurementResult]
    pending: int
    decision: Optional[SchedulerDecision]


class AbortResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunListResponse(BaseModel):
    runs: List[RunRecord]


@app.post("/experiments", response_model=SubmitResponse)
async def submit_experiment(plan: ExperimentPlan) -> SubmitResponse:
    run_id = deterministic_run_id(plan.dict())
    if STATE.get(run_id):
        raise HTTPException(status_code=400, detail="Run already exists")

    instrument = _build_instrument(plan)
    preflight = instrument.preflight(plan)
    if not preflight.ok:
        raise HTTPException(status_code=400, detail=f"Preflight failed: {preflight.message}")

    now = datetime.utcnow()
    plans_dir = DATA_DIR / "data" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"{run_id}.json"
    plan_path.write_text(json.dumps(plan.dict(), indent=2))

    events_dir = DATA_DIR / "data" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    results_path = str(DATA_DIR / "data" / "results")
    logs_path = str(DATA_DIR / "data" / "logs" / f"{run_id}.log")

    record = RunRecord(
        run_id=run_id,
        plan=plan,
        status=RunStatus.SUBMITTED,
        created_at=now,
        updated_at=now,
        backend="real" if feature_flag("REAL_DEVICE") else "sim",
        resource=os.getenv("INSTRUMENT_RESOURCE", "SIM"),
        plan_path=str(plan_path),
        events_path=str(events_dir),
        results_path=results_path,
        logs_path=logs_path,
    )

    run_state = STATE.create_run(record, instrument)
    run_state.log("Plan submitted")
    _emit_event(run_state, "Submitted", {"plan": plan.dict()})
    _persist_record(run_state.record)
    record.status = RunStatus.QUEUED
    record.updated_at = datetime.utcnow()
    return SubmitResponse(run_id=run_id, status=record.status)


@app.get("/experiments/{run_id}", response_model=RunStatusResponse)
async def get_experiment(run_id: str) -> RunStatusResponse:
    state = STATE.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatusResponse(
        record=state.record,
        completed=state.completed,
        pending=len(state.pending),
        decision=state.last_decision,
    )


@app.get("/runs", response_model=RunListResponse)
async def list_runs() -> RunListResponse:
    return RunListResponse(runs=STATE.list_runs())


@app.post("/actions/run-next", response_model=SchedulerDecision)
async def run_next(req: RunNextRequest) -> SchedulerDecision:
    state = STATE.get(req.run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    if not state.pending:
        raise HTTPException(status_code=400, detail="No pending tasks")

    tasks_with_duration = [(task, state.instrument.estimate_duration(task)) for task in state.pending]
    decision = state.scheduler.propose_next(tasks_with_duration)
    if decision is None:
        raise HTTPException(status_code=400, detail="Scheduler returned no tasks")
    state.last_decision = decision
    _emit_event(state, "Scheduled", {"decision": decision.dict()})
    state.record.eig_history.append(decision.eig_per_hour)
    if state.record.status != RunStatus.RUNNING:
        state.record.status = RunStatus.RUNNING
    task = next(t for t in state.pending if t.id == decision.task_id)
    state.pending.remove(task)
    start = datetime.utcnow()
    state.log(f"Starting task {task.id}")
    _emit_event(state, "Started", {"task": task.dict()})

    loop = asyncio.get_running_loop()
    measurement = await loop.run_in_executor(
        None, lambda: state.instrument.execute(task, state.cancel_token)
    )

    duration_s = (datetime.utcnow() - start).total_seconds()
    summary = {"duration_s": duration_s, "points": len(measurement.data.get("points", []))}

    if measurement.data.get("points"):
        parquet_path = state.parquet_writer.write_records(
            state.record.run_id,
            task.id,
            measurement.data["points"],
        )
    else:
        parquet_path = DATA_DIR / "data" / "results" / state.record.run_id / f"{task.id}.json"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text(json.dumps(measurement.data))

    result = MeasurementResult(
        task_id=task.id,
        started_at=start,
        finished_at=datetime.utcnow(),
        data_path=str(parquet_path),
        summary=summary,
    )
    state.completed.append(result)
    state.scheduler.update_measurement(result)
    _emit_event(state, "Measurement", {"result": result.dict()})
    state.record.predictive_variance.append(
        decision.candidate_scores[0]["posterior_variance"]
    )

    if not state.pending:
        state.record.status = RunStatus.COMPLETED
        _emit_event(state, "Completed", {"run_id": state.record.run_id})

    state.record.updated_at = datetime.utcnow()
    state.log(f"Finished task {task.id}")
    _persist_record(state.record)
    return decision


@app.post("/abort", response_model=AbortResponse)
async def abort_run(req: AbortRequest) -> AbortResponse:
    reason = req.reason or "operator request"
    aborted: Optional[str] = None
    for run_id, state in list(STATE.runs.items()):
        if state.record.status == RunStatus.RUNNING:
            state.cancel_token.set()
            state.record.status = RunStatus.ABORTED
            state.record.updated_at = datetime.utcnow()
            state.log(f"Abort issued: {reason}")
            _emit_event(state, "Aborted", {"reason": reason})
            _persist_record(state.record)
            aborted = run_id
    if not aborted:
        raise HTTPException(status_code=400, detail="No running run to abort")
    return AbortResponse(run_id=aborted, status=RunStatus.ABORTED)


@app.get("/provenance/{run_id}", response_model=ProvenanceGraph)
async def get_provenance(run_id: str) -> ProvenanceGraph:
    state = STATE.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    nodes = [
        {
            "id": f"plan-{run_id}",
            "type": "Plan",
            "label": state.record.plan.name,
            "data": state.record.plan.dict(),
            "parents": [],
        }
    ]
    for result in state.completed:
        nodes.append(
            {
                "id": f"task-{result.task_id}",
                "type": "Task",
                "label": result.task_id,
                "data": result.summary,
                "parents": [f"plan-{run_id}"],
            }
        )
        nodes.append(
            {
                "id": f"measurement-{result.task_id}",
                "type": "Measurement",
                "label": f"Measurement {result.task_id}",
                "data": {"path": result.data_path},
                "parents": [f"task-{result.task_id}"],
            }
        )
    return ProvenanceGraph(run_id=run_id, nodes=nodes)


@app.get("/experiments/{run_id}/logs")
async def stream_logs(run_id: str) -> StreamingResponse:
    state = STATE.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_stream():
        last_index = 0
        while True:
            await asyncio.sleep(1)
            if last_index < len(state.logs):
                chunk = "\n".join(state.logs[last_index:])
                last_index = len(state.logs)
                yield f"data: {chunk}\n\n"
            if state.record.status in {RunStatus.COMPLETED, RunStatus.ABORTED}:
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics() -> Response:
    lines = ["# HELP eig_per_hour Scheduler scores", "# TYPE eig_per_hour gauge"]
    for run in STATE.list_runs():
        for idx, value in enumerate(run.eig_history):
            lines.append(f"eig_per_hour{{run='{run.run_id}',step='{idx}'}} {value}")
    body = "\n".join(lines)
    return Response(content=body, media_type="text/plain")


def _build_instrument(plan: ExperimentPlan):
    real = feature_flag("REAL_DEVICE", default=False)
    if real:
        safety = plan.safety_limits
        if safety is None:
            raise HTTPException(status_code=400, detail="Safety limits required for real device")
        resource = os.getenv("INSTRUMENT_RESOURCE")
        if not resource:
            raise HTTPException(status_code=400, detail="INSTRUMENT_RESOURCE not configured")
        return build_real_device(resource, safety)
    return SimDevice(plan.instrument, seed=42, output_dir=DATA_DIR / "data" / "artifacts")


def _emit_event(state, event_type: str, payload: Dict[str, Any]) -> None:
    event = {
        "run_id": state.record.run_id,
        "event_id": f"evt-{datetime.utcnow().strftime('%H%M%S%f')}",
        "time": datetime.utcnow().isoformat(),
        "type": event_type,
        "payload": payload,
    }
    path = state.event_log.append(state.record.run_id, event)
    state.record.events_path = str(path)
    _persist_record(state.record)


def _persist_record(record: RunRecord) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{record.run_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(record.json())
