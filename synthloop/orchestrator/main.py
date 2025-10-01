from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
try:
    from sse_starlette.sse import EventSourceResponse
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from fastapi.responses import StreamingResponse

    class EventSourceResponse(StreamingResponse):
        def __init__(self, generator):
            super().__init__(generator, media_type="text/event-stream")

from .adapters.base import Rig
from .adapters.real import RealRig
from .adapters.sim import SimRig
from .models.plan import Outcome, RunCreate, RunStatus, Step, SynthesisPlan
from .qc.engine import run_qc
from .safety import interlocks
from .safety.watchdog import Watchdog
from .storage import db
from .storage.jsonl_eventlog import append_event
from .storage.parquet_writer import ARTIFACT_ROOT, write_bundle, write_step_telemetry

load_dotenv(dotenv_path=os.environ.get("ENV_FILE", None))

app = FastAPI(title="Synthesis Orchestrator")
security = HTTPBasic()


def get_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    user = os.environ.get("BASIC_AUTH_USER", "admin")
    password = os.environ.get("BASIC_AUTH_PASS", "changeme")
    if credentials.username != user or credentials.password != password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return credentials


def select_rig(plan: SynthesisPlan) -> Rig:
    real_device = os.environ.get("REAL_DEVICE", "false").lower() == "true"
    backend = plan.backend
    if backend == "real" or (backend == "auto" and real_device):
        return RealRig()
    return SimRig()


@dataclass
class RunContext:
    plan: SynthesisPlan
    rig: Rig
    status: str = "queued"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    log_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    interlock_history: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Optional[Outcome] = None


class RunManager:
    def __init__(self):
        self.runs: Dict[str, RunContext] = {}
        self.metrics = {"runs_total": 0, "runs_negative": 0}

    def create_run(self, plan: SynthesisPlan) -> str:
        rig = select_rig(plan)
        run_id = str(uuid.uuid4())
        ctx = RunContext(plan=plan, rig=rig)
        self.runs[run_id] = ctx
        db.init_db()
        db.insert_experiment(
            run_id,
            plan.model_dump_json(),
            plan.operator,
            "real" if isinstance(rig, RealRig) else "sim",
            isinstance(rig, RealRig),
        )
        self.log_event(run_id, "Submitted", {"plan_id": plan.plan_id})
        return run_id

    def log_event(self, run_id: str, event_type: str, payload: Dict[str, Any]):
        append_event(run_id, event_type, payload)
        db.record_event(run_id, event_type, payload)
        ctx = self.runs.get(run_id)
        if ctx:
            asyncio.run_coroutine_threadsafe(ctx.log_queue.put({"type": event_type, "payload": payload, "ts": datetime.utcnow().isoformat()}), loop)

    def start_run(self, run_id: str):
        ctx = self.runs.get(run_id)
        if not ctx:
            raise ValueError("Run not found")
        if ctx.status in {"running"}:
            raise ValueError("Run already started")
        thread = threading.Thread(target=self._execute_run, args=(run_id, ctx), daemon=True)
        thread.start()

    def abort_run(self, run_id: str):
        ctx = self.runs.get(run_id)
        if not ctx:
            raise ValueError("Run not found")
        ctx.cancel_event.set()
        self.log_event(run_id, "AbortRequested", {})

    def _execute_run(self, run_id: str, ctx: RunContext):
        ctx.status = "running"
        ctx.started_at = datetime.utcnow()
        self.log_event(run_id, "Started", {})
        try:
            interlock_state = getattr(ctx.rig, "interlocks", {
                "EnclosureClosed": True,
                "EStopNotEngaged": True,
                "VentilationOn": True,
                "ScaleHealthy": True,
                "PowerOK": True,
            })
            statuses = interlocks.verify(ctx.plan, interlock_state)
            ctx.interlock_history.extend([s.__dict__ for s in statuses])
            failed = [s for s in statuses if not s.ok]
            if failed:
                raise RuntimeError(f"Interlock failure: {[s.name for s in failed]}")
            self.log_event(run_id, "Preflight", {"statuses": [s.__dict__ for s in statuses]})
            run_dir = ARTIFACT_ROOT / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "plan.json").write_text(ctx.plan.model_dump_json(indent=2))
            for idx, step in enumerate(ctx.plan.steps):
                if ctx.cancel_event.is_set():
                    raise RuntimeError("Run aborted")
                self.log_event(run_id, "StepStarted", {"index": idx, "type": step.type})
                watchdog = Watchdog(
                    timeout_s=float(os.environ.get("WATCHDOG_TIMEOUT_S", 5)),
                    on_timeout=lambda: ctx.cancel_event.set(),
                )
                result = ctx.rig.execute_step(step, ctx.cancel_event)
                watchdog.stop()
                telemetry_path = write_step_telemetry(run_id, idx, result["telemetry"])
                ctx.step_results.append({
                    "step": step,
                    "status": result["status"],
                    "detail": result.get("detail", {}),
                    "telemetry_path": str(telemetry_path),
                })
                step_id = db.insert_step(
                    run_id,
                    step.type,
                    step.model_dump_json(),
                    result["started_at"],
                    result["ended_at"],
                    result["status"],
                    result["detail"].get("error") if isinstance(result.get("detail"), dict) else None,
                )
                db.insert_measurement(
                    run_id,
                    step_id,
                    str(telemetry_path),
                    json.dumps(result.get("detail", {}), default=str),
                )
                self.log_event(run_id, "StepResult", {"index": idx, "status": result["status"], "telemetry": str(telemetry_path)})
                if result["status"] == "failed":
                    raise RuntimeError(result["detail"].get("error", "Step failed"))
            run_dir = ARTIFACT_ROOT / run_id
            qc = run_qc(ctx.plan, ctx.step_results, run_dir, ctx.interlock_history)
            db.write_qc_report(run_id, qc["overall"], qc["rules"])
            qc_path = run_dir / "qc_report.json"
            qc_path.write_text(json.dumps(qc, default=str, indent=2))
            self.log_event(run_id, "QCCompleted", {"overall": qc["overall"]})
            success = qc["overall"] == "pass"
            failure_mode = None if success else "QCFail"
            outcome = Outcome(success=success, failure_mode=failure_mode, notes=None, evidence={"qc": qc})
            db.write_outcome(run_id, outcome.success, outcome.failure_mode, outcome.notes, outcome.evidence)
            ctx.outcome = outcome
            if not success:
                self.metrics["runs_negative"] += 1
            ctx.status = "completed" if success else "failed"
            self.log_event(run_id, "OutcomeRecorded", outcome.model_dump())
        except Exception as exc:  # noqa: BLE001
            failure_mode = "Abort" if ctx.cancel_event.is_set() else "InterlockViolation"
            if "overshoot" in str(exc).lower():
                failure_mode = "OverTemp"
            outcome = Outcome(success=False, failure_mode=failure_mode, notes=str(exc), evidence={"exception": str(exc)})
            run_dir = ARTIFACT_ROOT / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            qc = run_qc(
                ctx.plan,
                ctx.step_results,
                run_dir,
                ctx.interlock_history,
            )
            # Force QC failure on aborts or interlock errors
            extra_rule = {
                "name": "AbortHygiene" if ctx.cancel_event.is_set() else "Interlock",
                "status": "fail",
                "evidence": {"reason": str(exc)},
            }
            qc["rules"].append(extra_rule)
            qc["overall"] = "fail"
            db.write_qc_report(run_id, qc["overall"], qc["rules"])
            (run_dir / "qc_report.json").write_text(json.dumps(qc, default=str, indent=2))
            try:
                db.write_outcome(run_id, outcome.success, outcome.failure_mode, outcome.notes, outcome.evidence)
            except ValueError:
                pass
            ctx.outcome = outcome
            ctx.status = "aborted" if ctx.cancel_event.is_set() else "failed"
            self.metrics["runs_negative"] += 1
            self.log_event(run_id, "OutcomeRecorded", outcome.model_dump())
        finally:
            ctx.completed_at = datetime.utcnow()
            self.metrics["runs_total"] += 1
            self.log_event(run_id, "Completed", {"status": ctx.status})

    def list_runs(self, outcome_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        experiments = db.list_experiments()
        outcomes = db.list_outcomes()
        runs = []
        for exp in experiments:
            exp_dict = dict(exp)
            out = outcomes.get(exp_dict["run_id"])
            success = bool(out["success"]) if out else None
            if outcome_filter == "negative" and success is True:
                continue
            if outcome_filter == "positive" and (success is not True):
                continue
            runs.append(
                {
                    "run_id": exp_dict["run_id"],
                    "operator": exp_dict.get("operator"),
                    "backend": exp_dict.get("backend"),
                    "real_device": bool(exp_dict.get("real_device")),
                    "outcome": success,
                    "failure_mode": out["failure_mode"] if out else None,
                }
            )
        return runs


run_manager = RunManager()
loop = asyncio.new_event_loop()


def _run_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()


threading.Thread(target=_run_loop, daemon=True).start()


@app.post("/synthesis/plans", dependencies=[Depends(get_credentials)])
def create_plan(payload: RunCreate):
    run_id = run_manager.create_run(payload.plan)
    return {"run_id": run_id}


@app.post("/synthesis/runs/{run_id}/start", dependencies=[Depends(get_credentials)])
def start_run(run_id: str):
    try:
        run_manager.start_run(run_id)
        return {"status": "started"}
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/synthesis/runs/{run_id}/abort", dependencies=[Depends(get_credentials)])
def abort_run(run_id: str):
    try:
        run_manager.abort_run(run_id)
        return {"status": "aborting"}
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/synthesis/runs/{run_id}", dependencies=[Depends(get_credentials)])
def get_run(run_id: str):
    ctx = run_manager.runs.get(run_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": run_id,
        "status": ctx.status,
        "started_at": ctx.started_at,
        "completed_at": ctx.completed_at,
        "outcome": ctx.outcome.model_dump() if ctx.outcome else None,
    }


@app.get("/synthesis/runs", dependencies=[Depends(get_credentials)])
def list_runs(outcome: str = "all"):
    outcome_filter = None if outcome == "all" else outcome
    return run_manager.list_runs(outcome_filter)


@app.get("/provenance/{run_id}", dependencies=[Depends(get_credentials)])
def provenance(run_id: str):
    run_dir = ARTIFACT_ROOT / run_id
    events_dir = Path(os.environ.get("EVENT_LOG_DIR", "./data/events"))
    events = []
    for day_dir in events_dir.glob("*"):
        path = day_dir / f"{run_id}.jsonl"
        if path.exists():
            events.extend([json.loads(line) for line in path.read_text().splitlines() if line])
    measurements = []
    step_records = db.list_steps(run_id)
    for rec in step_records:
        rec_dict = dict(rec)
        measurements.append(
            {
                "step": rec_dict.get("name"),
                "status": rec_dict.get("status"),
                "started_at": rec_dict.get("started_at"),
                "ended_at": rec_dict.get("ended_at"),
            }
        )
    return {"events": events, "measurements": measurements, "artifacts": [str(p) for p in run_dir.glob("*")]}


@app.get("/qc/{run_id}", dependencies=[Depends(get_credentials)])
def qc_report(run_id: str):
    record = db.get_qc_report(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="QC not ready")
    return {
        "overall": record["overall_status"],
        "rules": json.loads(record["rules_summary_json"]),
        "created_at": record["created_at"],
    }


@app.get("/bundles/{run_id}", dependencies=[Depends(get_credentials)])
def bundle(run_id: str):
    run_dir = ARTIFACT_ROOT / run_id
    files = {"qc_report.json": run_dir / "qc_report.json"}
    plan_path = run_dir / "plan.json"
    ctx = run_manager.runs.get(run_id)
    if ctx:
        plan_path.write_text(ctx.plan.model_dump_json(indent=2))
        files["plan.json"] = plan_path
    events_dir = Path(os.environ.get("EVENT_LOG_DIR", "./data/events"))
    for day_dir in events_dir.glob("*"):
        path = day_dir / f"{run_id}.jsonl"
        if path.exists():
            files[f"events/{path.name}"] = path
    bundle_path = write_bundle(run_id, files)
    return Response(content=bundle_path.read_bytes(), media_type="application/zip")


@app.get("/sse/{run_id}", dependencies=[Depends(get_credentials)])
async def stream(run_id: str):
    ctx = run_manager.runs.get(run_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="run not found")

    async def event_generator():
        while True:
            message = await ctx.log_queue.get()
            yield {"event": "log", "data": json.dumps(message)}

    return EventSourceResponse(event_generator())


@app.get("/metrics")
def metrics():
    total = run_manager.metrics["runs_total"]
    negative = run_manager.metrics["runs_negative"]
    lines = [
        "# HELP runs_total Total runs",
        "# TYPE runs_total counter",
        f"runs_total {total}",
        "# HELP runs_negative Total negative runs",
        "# TYPE runs_negative counter",
        f"runs_negative {negative}",
    ]
    return Response("\n".join(lines), media_type="text/plain")


