from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator

StepType = Literal["Dispense", "Mix", "Heat", "Cool", "Collect"]


class Step(BaseModel):
    type: StepType

    reagent: Optional[str] = None
    mass_g: Optional[float] = Field(None, gt=0)
    duration_s: Optional[int] = Field(None, gt=0)
    rpm: Optional[int] = Field(None, ge=0)
    target_C: Optional[float] = None
    ramp_C_per_min: Optional[float] = None
    hold_min: Optional[float] = None
    sample_id: Optional[str] = None

    @validator("reagent", always=True)
    def reagent_required(cls, v, values):
        if values.get("type") == "Dispense" and not v:
            raise ValueError("Dispense step requires reagent")
        return v

    @validator("mass_g", always=True)
    def mass_required(cls, v, values):
        if values.get("type") == "Dispense" and v is None:
            raise ValueError("Dispense step requires mass_g")
        return v

    @validator("duration_s", always=True)
    def mix_duration(cls, v, values):
        if values.get("type") == "Mix" and v is None:
            raise ValueError("Mix step requires duration_s")
        return v

    @validator("rpm", always=True)
    def mix_rpm(cls, v, values):
        if values.get("type") == "Mix" and v is None:
            raise ValueError("Mix step requires rpm")
        return v

    @validator("target_C", always=True)
    def temperature_required(cls, v, values):
        if values.get("type") in {"Heat", "Cool"} and v is None:
            raise ValueError("Temperature target required")
        return v

    @validator("sample_id", always=True)
    def collect_sample(cls, v, values):
        if values.get("type") == "Collect" and not v:
            raise ValueError("Collect step requires sample_id")
        return v


class CalibrationRef(BaseModel):
    scale_id: str
    temp_probe_id: str
    last_calibrated_scale: datetime = Field(alias="last_calibrated", default_factory=datetime.utcnow)
    last_calibrated_temp: datetime = Field(alias="last_calibrated_temp", default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


class SynthesisPlan(BaseModel):
    plan_id: str
    operator: str
    operator_ack: bool
    backend: Literal["auto", "real", "sim"] = "auto"
    sample: dict
    reagents: List[dict]
    steps: List[Step]
    calibration_refs: dict
    notes: Optional[str] = None

    @validator("operator_ack")
    def operator_ack_required(cls, v):
        if not v:
            raise ValueError("operator_ack must be true")
        return v

    @validator("sample")
    def sample_fields(cls, v):
        for field in ("id", "batch_g", "tolerance_g"):
            if field not in v:
                raise ValueError(f"sample.{field} required")
        return v

    @validator("calibration_refs")
    def cal_refs(cls, v):
        for field in ("scale_id", "temp_probe_id", "last_calibrated", "last_calibrated_temp"):
            if field not in v:
                raise ValueError(f"calibration_refs.{field} required")
        return v


class RunCreate(BaseModel):
    plan: SynthesisPlan
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


class RunStatus(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed", "aborted"]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class StepResult(BaseModel):
    step: Step
    started_at: datetime
    ended_at: datetime
    status: Literal["success", "failed", "aborted"]
    telemetry_path: str
    summary: dict


class QCReport(BaseModel):
    overall: Literal["pass", "warn", "fail"]
    rules: List[dict]
    created_at: datetime


class Outcome(BaseModel):
    success: bool
    failure_mode: Optional[str]
    notes: Optional[str]
    evidence: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

