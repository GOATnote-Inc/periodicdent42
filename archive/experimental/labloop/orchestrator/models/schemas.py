from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class InstrumentType(str, Enum):
    XRD = "xrd"
    TRANSPORT = "transport"


class SafetyLimits(BaseModel):
    max_current: Optional[float] = Field(default=None, description="Ampere")
    max_voltage: Optional[float] = Field(default=None, description="Volt")
    max_temperature: Optional[float] = Field(default=None, description="Kelvin")
    max_scan_rate: Optional[float] = Field(default=None, description="Degrees per second")

    def ensure_complete(self) -> None:
        missing = [
            name
            for name, value in (
                ("max_current", self.max_current),
                ("max_voltage", self.max_voltage),
                ("max_temperature", self.max_temperature),
                ("max_scan_rate", self.max_scan_rate),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Missing safety limits: {', '.join(missing)}")


class XRDTask(BaseModel):
    two_theta_start: float = Field(..., ge=0, le=180)
    two_theta_end: float = Field(..., ge=0, le=180)
    step: float = Field(..., gt=0, le=5)
    dwell_time: float = Field(..., gt=0, description="Seconds per step")

    @validator("two_theta_end")
    def validate_range(cls, v: float, values: Dict[str, Any]) -> float:
        start = values.get("two_theta_start")
        if start is not None and v <= start:
            raise ValueError("two_theta_end must be greater than two_theta_start")
        return v


class TransportTask(BaseModel):
    current_start: float
    current_end: float
    num_points: int = Field(..., gt=1, le=1000)
    hold_time: float = Field(..., gt=0, description="Seconds per point")
    temperature: Optional[float] = Field(default=None)


class TaskType(str, Enum):
    XRD = "xrd"
    TRANSPORT = "transport"


class Task(BaseModel):
    id: str
    type: TaskType
    parameters: Dict[str, Any]
    expected_duration_s: Optional[float] = None


class ExperimentPlan(BaseModel):
    name: str
    operator: str
    instrument: InstrumentType
    tasks: List[Task]
    safety_limits: Optional[SafetyLimits] = None
    operator_ack: bool = False

    @validator("tasks")
    def validate_tasks(cls, v: List[Task], values: Dict[str, Any]) -> List[Task]:
        instrument = values.get("instrument")
        for task in v:
            if instrument == InstrumentType.XRD and task.type != TaskType.XRD:
                raise ValueError("XRD instrument requires XRD tasks")
            if instrument == InstrumentType.TRANSPORT and task.type != TaskType.TRANSPORT:
                raise ValueError("Transport instrument requires transport tasks")
        return v


class RunStatus(str, Enum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"


class RunRecord(BaseModel):
    run_id: str
    plan: ExperimentPlan
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    backend: str
    resource: str
    plan_path: str
    events_path: str
    results_path: str
    logs_path: str
    eig_history: List[float] = Field(default_factory=list)
    predictive_variance: List[float] = Field(default_factory=list)


class ProvenanceNode(BaseModel):
    id: str
    type: str
    label: str
    data: Dict[str, Any]
    parents: List[str]


class ProvenanceGraph(BaseModel):
    run_id: str
    nodes: List[ProvenanceNode]


class SchedulerDecision(BaseModel):
    task_id: str
    eig_bits: float
    duration_hours: float
    eig_per_hour: float
    rationale: str
    candidate_scores: List[Dict[str, Any]]


class MeasurementResult(BaseModel):
    task_id: str
    started_at: datetime
    finished_at: datetime
    data_path: str
    summary: Dict[str, Any]


class AbortRequest(BaseModel):
    reason: Optional[str] = None


class RunNextRequest(BaseModel):
    run_id: str
