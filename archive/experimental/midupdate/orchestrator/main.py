from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from orchestrator.agent.constraint_checker import ConstraintChecker
from orchestrator.agent.planner import GlassBoxPlanner, PlannerConfig, validate_plan
from orchestrator.scheduler.surrogate import GPSurrogate
from orchestrator.storage.logger import EventLogger

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_REGISTRY = BASE_DIR / "training" / "model_registry"
LOG_DIR = BASE_DIR / "orchestrator" / "storage" / "artifacts"


class PlanRequest(BaseModel):
    context: Dict[str, Any]
    objective: str
    constraints: Dict[str, float]
    candidate_space: Dict[str, Any]
    campaign_id: Optional[str] = None
    step: int = 0


class PlanCritiqueRequest(BaseModel):
    plan: Dict[str, Any]


app = FastAPI(title="Glass-Box Planner Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
planner = GlassBoxPlanner(PlannerConfig(model_registry=MODEL_REGISTRY, dataset_dir=DATA_DIR))
constraint_checker = ConstraintChecker(env_path=str(BASE_DIR / ".env"))
surrogate = GPSurrogate()
event_logger = EventLogger(LOG_DIR)


@app.post("/plan/next")
def plan_next(request: PlanRequest) -> Dict[str, Any]:
    plan = planner.propose(
        context=request.context,
        objective=request.objective,
        constraints=request.constraints,
        candidate_space=request.candidate_space,
    )
    valid, issues = validate_plan(plan)
    if not valid:
        raise HTTPException(status_code=422, detail={"errors": issues})
    ok, repaired_plan, violations = constraint_checker.check(plan)
    event_logger.log_event(
        "plan_generated",
        {
            "campaign_id": request.campaign_id,
            "step": request.step,
            "plan": repaired_plan,
            "violations": [violation.__dict__ for violation in violations],
        },
    )
    response = {
        "plan": repaired_plan,
        "auto_repaired": not ok,
        "violations": [violation.__dict__ for violation in violations],
    }
    return response


@app.post("/plan/critique")
def plan_critique(request: PlanCritiqueRequest) -> Dict[str, Any]:
    valid, issues = validate_plan(request.plan)
    ok, repaired_plan, violations = constraint_checker.check(request.plan)
    critique = {
        "schema_valid": valid,
        "issues": issues,
        "repaired_plan": repaired_plan,
        "violations": [violation.__dict__ for violation in violations],
    }
    event_logger.log_event("plan_critiqued", critique)
    return critique


@app.get("/models/current")
def models_current() -> Dict[str, Any]:
    latest_dir = MODEL_REGISTRY / "latest"
    if not latest_dir.exists():
        raise HTTPException(status_code=404, detail="No model available")
    metrics_path = latest_dir / "metrics.json"
    config_path = latest_dir / "train_config.yaml"
    model_card_path = latest_dir / "model_card.md"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    config_text = config_path.read_text() if config_path.exists() else ""
    model_card = model_card_path.read_text() if model_card_path.exists() else ""
    return {
        "model_dir": str(latest_dir),
        "metrics": metrics,
        "train_config": config_text,
        "model_card": model_card,
    }


@app.get("/campaigns")
def campaign_list() -> Dict[str, Any]:
    results_path = BASE_DIR / "campaigns" / "results.json"
    if not results_path.exists():
        return {"campaigns": []}
    payload = json.loads(results_path.read_text())
    return {"campaigns": payload}


@app.get("/campaigns/{campaign_id}")
def campaign_detail(campaign_id: str) -> Dict[str, Any]:
    results_path = BASE_DIR / "campaigns" / "results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No campaign results available")
    payload = json.loads(results_path.read_text())
    for entry in payload:
        if entry.get("campaign") == campaign_id:
            return entry
    raise HTTPException(status_code=404, detail="Campaign not found")


@app.get("/events")
def stream_events() -> StreamingResponse:
    def event_generator() -> Generator[bytes, None, None]:
        events_path = event_logger.events_path
        last_size = 0
        while True:
            if events_path.exists():
                data = events_path.read_text()
                if len(data) != last_size:
                    chunk = data[last_size:]
                    last_size = len(data)
                    yield f"data: {chunk}\n\n".encode("utf-8")
                else:
                    yield b"data: {}\n\n"
            else:
                yield b"data: {}\n\n"
            time.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/abort")
def abort() -> Dict[str, str]:
    event_logger.log_event("abort", {"status": "triggered"})
    return {"status": "aborting"}
