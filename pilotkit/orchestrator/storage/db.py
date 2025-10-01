from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select, delete


class MetricRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    unit_id: str
    period: datetime
    cycle_time_s: float
    touch_time_s: float
    wait_time_s: float
    yield_ok: bool
    defect_code: Optional[str]


class FeedbackRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime
    step: Optional[str]
    severity: str
    tags: str
    text: str
    frustration: Optional[int]
    task_success: Optional[bool]
    time_on_task_s: Optional[int]
    theme: Optional[str]


class SurveyRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime
    respondent: Optional[str]
    sus_score: Optional[float]
    nps_score: Optional[int]
    csat_score: Optional[int]


class IterationPlanRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    generated_at: datetime
    title: str
    owner: str
    metric: str
    guardrail: str
    ice_score: float


class Database:
    def __init__(self, path: Path):
        self.engine = create_engine(f"sqlite:///{path}")
        SQLModel.metadata.create_all(self.engine)

    def add_metrics(self, rows: Iterable[MetricRow]) -> None:
        with Session(self.engine) as session:
            for row in rows:
                session.add(row)
            session.commit()

    def replace_metrics(self, rows: Iterable[MetricRow]) -> None:
        with Session(self.engine) as session:
            session.exec(delete(MetricRow))
            for row in rows:
                session.add(row)
            session.commit()

    def get_metrics(self, start: datetime | None = None, end: datetime | None = None):
        with Session(self.engine) as session:
            stmt = select(MetricRow)
            if start is not None:
                stmt = stmt.where(MetricRow.period >= start)
            if end is not None:
                stmt = stmt.where(MetricRow.period <= end)
            return session.exec(stmt).all()

    def add_feedback(self, rows: Iterable[FeedbackRow]) -> None:
        with Session(self.engine) as session:
            for row in rows:
                session.add(row)
            session.commit()

    def add_surveys(self, rows: Iterable[SurveyRow]) -> None:
        with Session(self.engine) as session:
            for row in rows:
                session.add(row)
            session.commit()

    def add_iteration_plan(self, rows: Iterable[IterationPlanRow]) -> None:
        with Session(self.engine) as session:
            for row in rows:
                session.add(row)
            session.commit()

    def list_iteration_plan(self) -> list[IterationPlanRow]:
        with Session(self.engine) as session:
            return session.exec(select(IterationPlanRow)).all()
