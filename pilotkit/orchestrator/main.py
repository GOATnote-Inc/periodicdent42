from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Dict, List

import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sse_starlette.sse import EventSourceResponse

from .adapters.http_adapter import HTTPAdapter
from .analytics.metrics import MetricEmitter, split_baseline_pilot
from .analytics.value_stream import summarize_metrics
from .feedback.processor import FeedbackProcessor
from .iteration.planner import export_plan, rank_backlog
from .models.schemas import (
    CandidateScore,
    CandidateScoreRequest,
    CandidateScoreResponse,
    FeedbackIngestRequest,
    IterationPlanFile,
    IterationPlanRequest,
    MetricsSummary,
    PilotConfig,
    ReportRequest,
    ReportResponse,
    SurveySubmission,
    SUSNPSResponse,
    WorkflowEvent,
)
from .playbook.generator import generate_playbook
from .report.generator import PilotReportGenerator
from .storage.db import Database, FeedbackRow, IterationPlanRow, MetricRow, SurveyRow
from sqlmodel import Session, select
from .storage.eventlog import EventLog
from .storage.parquet_writer import ParquetWriter

app = FastAPI(title="Pilot Orchestrator")
security = HTTPBasic()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / ".." / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVENT_LOG = EventLog(DATA_DIR / "events.jsonl")
PARQUET_WRITER = ParquetWriter(DATA_DIR / "metrics.parquet")
DATABASE = Database(DATA_DIR / "pilot.db")
CONFIG_STORE = DATA_DIR / "pilot-config.yaml"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLAYBOOK_DIR = REPORT_DIR
ITERATION_DIR = REPORT_DIR
WORKFLOW_CONFIG = Path(
    os.getenv(
        "WORKFLOW_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "workflow.sample.yaml",
    )
)

PII_ALLOWED = os.getenv("PII_ALLOWED", "false").lower() == "true"
processor = FeedbackProcessor(pii_allowed=PII_ALLOWED)
http_adapter = HTTPAdapter()

METRIC_QUEUE: asyncio.Queue = asyncio.Queue()


def _load_steps() -> List[str]:
    if not WORKFLOW_CONFIG.exists():
        return ["queue", "work", "review", "done"]
    data = yaml.safe_load(WORKFLOW_CONFIG.read_text())
    return [step["name"] for step in data.get("steps", [])]


STEPS = _load_steps()


def get_auth(credentials: HTTPBasicCredentials = Depends(security)):
    user = os.getenv("AUTH_BASIC_USER", "pilot")
    password = os.getenv("AUTH_BASIC_PASS")
    if not password:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AUTH_BASIC_PASS environment variable must be set",
            headers={"WWW-Authenticate": "Basic"},
        )
    if credentials.username != user or credentials.password != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Basic"},
            detail="Invalid credentials",
        )
    return True


def load_events() -> List[WorkflowEvent]:
    return EVENT_LOG.read_all()


def compute_metric_records() -> List[dict]:
    events = load_events()
    if not events:
        return []
    emitter = MetricEmitter(STEPS)
    return emitter.emit(events)


@app.post("/pilot/candidates/score", response_model=CandidateScoreResponse)
async def score_candidates(
    payload: CandidateScoreRequest, auth: bool = Depends(get_auth)
):
    weights = payload.weights.normalized()
    ranked = []
    for candidate in payload.candidates:
        score = (
            candidate.exec_sponsor * weights.exec_sponsor
            + candidate.data_access * weights.data_access
            + candidate.workflow_speed * weights.workflow_speed
            + candidate.potential_value * weights.potential_value
            + (5 - candidate.risk) * weights.risk
            + candidate.champion * weights.champion
        )
        rationale = (
            f"High sponsor support" if candidate.exec_sponsor > 3 else "Needs stronger sponsor"
        )
        ranked.append(CandidateScore(candidate=candidate, score=float(round(score, 3)), rationale=rationale))
    ranked.sort(key=lambda c: c.score, reverse=True)
    manual_override_applied = False
    if payload.manual_override:
        for idx, cand in enumerate(ranked):
            if cand.candidate.name == payload.manual_override:
                ranked.insert(0, ranked.pop(idx))
                manual_override_applied = True
                break
    return CandidateScoreResponse(ranked=ranked, manual_override_applied=manual_override_applied)


@app.post("/pilot/config", response_model=PilotConfig)
async def update_config(config: PilotConfig, auth: bool = Depends(get_auth)):
    import yaml

    CONFIG_STORE.write_text(yaml.safe_dump(config.dict(), sort_keys=False))
    generate_playbook(CONFIG_STORE, PLAYBOOK_DIR)
    return config


@app.post("/ingest/event")
async def ingest_event(event: WorkflowEvent, auth: bool = Depends(get_auth)):
    EVENT_LOG.append([event])
    http_adapter.add_event(event)
    records = compute_metric_records()
    PARQUET_WRITER.write(records)
    metric_rows = [
        MetricRow(
            unit_id=r["unit_id"],
            period=datetime.fromisoformat(r["period"]),
            cycle_time_s=r["cycle_time_s"],
            touch_time_s=r["touch_time_s"],
            wait_time_s=r["wait_time_s"],
            yield_ok=r["yield_ok"],
            defect_code=r["defect_code"],
        )
        for r in records
    ]
    DATABASE.replace_metrics(metric_rows)
    summary = summarize_metrics(records)
    await METRIC_QUEUE.put(summary)
    return {"status": "ok"}


@app.get("/metrics/summary", response_model=MetricsSummary)
async def metrics_summary(
    window: str = Query("7d"), auth: bool = Depends(get_auth)
):
    records = compute_metric_records()
    if window.endswith("d"):
        days = int(window.rstrip("d"))
        cutoff = datetime.utcnow() - timedelta(days=days)
        records = [r for r in records if r["end_ts"] >= cutoff]
    summary = summarize_metrics(records)
    return MetricsSummary(window=window, **summary)


@app.get("/metrics/baseline")
async def metrics_baseline(auth: bool = Depends(get_auth)):
    records = compute_metric_records()
    split = datetime.utcnow() - timedelta(days=7)
    baseline, _ = split_baseline_pilot(records, split)
    return baseline


@app.get("/metrics/pilot")
async def metrics_pilot(auth: bool = Depends(get_auth)):
    records = compute_metric_records()
    split = datetime.utcnow() - timedelta(days=7)
    _, pilot = split_baseline_pilot(records, split)
    return pilot


@app.get("/metrics/baseline-summary", response_model=MetricsSummary)
async def metrics_baseline_summary(auth: bool = Depends(get_auth)):
    records = compute_metric_records()
    split = datetime.utcnow() - timedelta(days=7)
    baseline, _ = split_baseline_pilot(records, split)
    summary = summarize_metrics(baseline)
    return MetricsSummary(window="baseline", **summary)


@app.get("/metrics/pilot-summary", response_model=MetricsSummary)
async def metrics_pilot_summary(auth: bool = Depends(get_auth)):
    records = compute_metric_records()
    split = datetime.utcnow() - timedelta(days=7)
    _, pilot = split_baseline_pilot(records, split)
    summary = summarize_metrics(pilot)
    return MetricsSummary(window="pilot", **summary)


@app.post("/feedback")
async def submit_feedback(payload: FeedbackIngestRequest, auth: bool = Depends(get_auth)):
    themes = processor.aggregate_insights(payload.items)
    interview_guide = processor.generate_interview_guide([t["theme"] for t in themes])
    db_rows = []
    for item in payload.items:
        tags = processor.auto_tag(item)
        db_rows.append(
            FeedbackRow(
                ts=item.ts,
                step=item.step,
                severity=item.severity,
                tags=",".join(tags),
                text=processor._redact(item.text),
                frustration=item.frustration,
                task_success=item.task_success,
                time_on_task_s=item.time_on_task_s,
                theme=tags[0] if tags else None,
            )
        )
    DATABASE.add_feedback(db_rows)
    return {"themes": themes, "count": len(payload.items), "interview_guide": interview_guide}


@app.post("/survey/submit")
async def submit_survey(submission: SurveySubmission, auth: bool = Depends(get_auth)):
    DATABASE.add_surveys(
        [
            SurveyRow(
                ts=submission.ts,
                respondent=submission.respondent,
                sus_score=submission.sus_score,
                nps_score=submission.nps_score,
                csat_score=submission.csat_score,
            )
        ]
    )
    return {"status": "recorded"}


@app.get("/survey/summary", response_model=SUSNPSResponse)
async def survey_summary(auth: bool = Depends(get_auth)):
    with Session(DATABASE.engine) as session:
        rows = session.exec(select(SurveyRow)).all()
    if not rows:
        return SUSNPSResponse(sus_avg=None, nps=None, csat_avg=None, trends=[])
    sus_scores = [r.sus_score for r in rows if r.sus_score is not None]
    csat_scores = [r.csat_score for r in rows if r.csat_score is not None]
    nps_scores = [r.nps_score for r in rows if r.nps_score is not None]
    sus_avg = sum(sus_scores) / len(sus_scores) if sus_scores else None
    csat_avg = sum(csat_scores) / len(csat_scores) if csat_scores else None
    promoters = len([score for score in nps_scores if score >= 9])
    detractors = len([score for score in nps_scores if score <= 6])
    total = len(nps_scores)
    nps = ((promoters - detractors) / total) * 100 if total else None
    trends = []
    if sus_scores:
        trends.append({"metric": "SUS", "values": sus_scores})
    if nps_scores:
        trends.append({"metric": "NPS", "values": nps_scores})
    if csat_scores:
        trends.append({"metric": "CSAT", "values": csat_scores})
    return SUSNPSResponse(
        sus_avg=float(sus_avg) if sus_avg is not None else None,
        nps=float(nps) if nps is not None else None,
        csat_avg=float(csat_avg) if csat_avg is not None else None,
        trends=[{"metric": t["metric"], "values": t["values"]} for t in trends],
    )


@app.post("/iteration/plan", response_model=IterationPlanFile)
async def create_iteration_plan(
    request: IterationPlanRequest, auth: bool = Depends(get_auth)
):
    plan = rank_backlog(request)
    files = export_plan(plan, ITERATION_DIR)
    DATABASE.add_iteration_plan(
        [
            IterationPlanRow(
                generated_at=plan.generated_at,
                title=item.title,
                owner=item.owner,
                metric=item.metric,
                guardrail=item.guardrail,
                ice_score=item.ice_score,
            )
            for item in plan.items
        ]
    )
    return IterationPlanFile(markdown_path=str(files["markdown"]), json_path=str(files["json"]))


@app.post("/report/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest, auth: bool = Depends(get_auth)):
    records = compute_metric_records()
    baseline, pilot = split_baseline_pilot(records, request.pilot_start)
    generator = PilotReportGenerator(REPORT_DIR)
    result = generator.generate(baseline, pilot, request.title)
    return ReportResponse(markdown_path=str(result["markdown"]), chart_path=str(result["chart"]))


@app.get("/metrics/stream")
async def metrics_stream(request: Request, auth: bool = Depends(get_auth)):
    async def event_publisher() -> AsyncGenerator[Dict[str, str], None]:
        while True:
            if await request.is_disconnected():
                break
            summary = await METRIC_QUEUE.get()
            yield {"event": "metrics", "data": json.dumps(summary)}

    return EventSourceResponse(event_publisher())
