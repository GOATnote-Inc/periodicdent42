from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    name: str
    exec_sponsor: float = Field(ge=0, le=5)
    data_access: float = Field(ge=0, le=5)
    workflow_speed: float = Field(ge=0, le=5)
    potential_value: float = Field(ge=0, le=5)
    risk: float = Field(ge=0, le=5)
    champion: float = Field(ge=0, le=5)
    notes: Optional[str] = None


class ScoreWeights(BaseModel):
    exec_sponsor: float = 0.2
    data_access: float = 0.2
    workflow_speed: float = 0.2
    potential_value: float = 0.2
    risk: float = 0.1
    champion: float = 0.1

    def normalized(self) -> "ScoreWeights":
        total = (
            self.exec_sponsor
            + self.data_access
            + self.workflow_speed
            + self.potential_value
            + self.risk
            + self.champion
        )
        if total == 0:
            return self
        return ScoreWeights(
            exec_sponsor=self.exec_sponsor / total,
            data_access=self.data_access / total,
            workflow_speed=self.workflow_speed / total,
            potential_value=self.potential_value / total,
            risk=self.risk / total,
            champion=self.champion / total,
        )


class CandidateScore(BaseModel):
    candidate: Candidate
    score: float
    rationale: str


class CandidateScoreRequest(BaseModel):
    weights: ScoreWeights
    candidates: List[Candidate]
    manual_override: Optional[str] = Field(
        None, description="Optional candidate name to force rank first"
    )


class CandidateScoreResponse(BaseModel):
    ranked: List[CandidateScore]
    manual_override_applied: bool = False


class PilotConfig(BaseModel):
    pilot_name: str
    partner: str
    scope: Dict[str, object]
    success_criteria: Dict[str, object]
    timeline: Dict[str, str]
    risks: List[str]
    comms_plan: Dict[str, str]
    legal: Dict[str, str]


class WorkflowEvent(BaseModel):
    ts: datetime
    unit_id: str
    step: str
    state: str
    attrs: Dict[str, Optional[str]] = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    window: str
    cycle_time_p50: float
    cycle_time_p90: float
    throughput_per_day: float
    wip: float
    yield_rate: float
    defects: Dict[str, int]
    units: int


class FeedbackItem(BaseModel):
    ts: datetime
    submitter: Optional[str] = None
    step: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    severity: str = Field(..., regex="^P[0-3]$")
    text: str
    frustration: Optional[int] = Field(None, ge=1, le=5)
    task_success: Optional[bool] = None
    time_on_task_s: Optional[int] = None
    screenshot: Optional[str] = None


class FeedbackIngestRequest(BaseModel):
    items: List[FeedbackItem]


class SurveySubmission(BaseModel):
    ts: datetime
    respondent: Optional[str]
    sus_score: Optional[float] = Field(None, ge=0, le=100)
    nps_score: Optional[int] = Field(None, ge=-100, le=100)
    csat_score: Optional[int] = Field(None, ge=1, le=5)
    notes: Optional[str] = None


class IterationBacklogItem(BaseModel):
    title: str
    description: str
    impact: float
    confidence: float
    effort: float
    ice_score: float
    owner: str
    eta_days: int
    metric: str
    guardrail: str


class IterationPlan(BaseModel):
    generated_at: datetime
    items: List[IterationBacklogItem]
    summary: str


class IterationPlanRequest(BaseModel):
    metrics_delta: Dict[str, float]
    top_feedback_themes: List[str]
    guardrails: Dict[str, str] = Field(default_factory=dict)


class ReportRequest(BaseModel):
    baseline_start: datetime
    baseline_end: datetime
    pilot_start: datetime
    pilot_end: datetime
    mde_pct: float
    sample_size: int
    title: str


class ReportResponse(BaseModel):
    markdown_path: str
    chart_path: str


class SurveyTrend(BaseModel):
    metric: str
    values: List[float]


class SUSNPSResponse(BaseModel):
    sus_avg: Optional[float]
    nps: Optional[float]
    csat_avg: Optional[float]
    trends: List[SurveyTrend]


class IterationPlanFile(BaseModel):
    markdown_path: str
    json_path: str
