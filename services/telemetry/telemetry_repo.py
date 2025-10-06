from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class RunModel(Base):
    __tablename__ = "telemetry_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    input_hash: Mapped[Optional[str]] = mapped_column(String(64))
    summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
    )
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"))


class EventModel(Base):
    __tablename__ = "telemetry_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36))
    sequence: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class MetricModel(Base):
    __tablename__ = "telemetry_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36))
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(32))
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column("metadata", JSON)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class ErrorModel(Base):
    __tablename__ = "telemetry_errors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36))
    message: Mapped[str] = mapped_column(String)
    details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class ArtifactModel(Base):
    __tablename__ = "telemetry_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36))
    name: Mapped[str] = mapped_column(String(64))
    uri: Mapped[Optional[str]] = mapped_column(String)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column("metadata", JSON)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


@dataclass
class TelemetryRun:
    id: str
    run_type: str
    status: str
    input_hash: Optional[str]
    summary: Optional[dict[str, Any]]
    created_at: dt.datetime
    updated_at: dt.datetime
    completed_at: Optional[dt.datetime]


@dataclass
class TelemetryEvent:
    id: int
    run_id: str
    sequence: int
    event_type: str
    payload: Optional[dict[str, Any]]
    created_at: dt.datetime


class TelemetryRepo:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_run(
        self,
        *,
        run_type: str,
        status: str,
        input_hash: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> TelemetryRun:
        with self.session_factory() as session:
            run = RunModel(
                run_type=run_type,
                status=status,
                input_hash=input_hash,
                summary=summary,
                created_at=dt.datetime.now(dt.timezone.utc),
                updated_at=dt.datetime.now(dt.timezone.utc),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return self._to_run(run)

    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
        completed_at: dt.datetime | None = None,
    ) -> TelemetryRun:
        with self.session_factory() as session:
            run = session.get(RunModel, run_id)
            if not run:
                raise KeyError(f"run {run_id} not found")
            run.status = status
            if summary is not None:
                run.summary = summary
            run.updated_at = dt.datetime.now(dt.timezone.utc)
            if completed_at:
                run.completed_at = completed_at
            session.commit()
            session.refresh(run)
            return self._to_run(run)

    def soft_delete(self, run_id: str) -> None:
        with self.session_factory() as session:
            run = session.get(RunModel, run_id)
            if not run:
                return
            run.is_deleted = True
            run.updated_at = dt.datetime.now(dt.timezone.utc)
            session.commit()

    def list_runs(self, *, limit: int = 50, status: str | None = None) -> List[TelemetryRun]:
        with self.session_factory() as session:
            stmt = select(RunModel).where(RunModel.is_deleted.is_(False)).order_by(RunModel.created_at.desc())
            if status:
                stmt = stmt.where(RunModel.status == status)
            stmt = stmt.limit(limit)
            results = session.execute(stmt).scalars().all()
            return [self._to_run(row) for row in results]

    def get_run(self, run_id: str) -> TelemetryRun | None:
        with self.session_factory() as session:
            run = session.get(RunModel, run_id)
            if not run or run.is_deleted:
                return None
            return self._to_run(run)

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> TelemetryEvent:
        with self.session_factory() as session:
            sequence_stmt = (
                select(EventModel.sequence)
                .where(EventModel.run_id == run_id)
                .order_by(EventModel.sequence.desc())
                .limit(1)
            )
            last_sequence = session.execute(sequence_stmt).scalar_one_or_none() or 0
            event = EventModel(
                run_id=run_id,
                sequence=last_sequence + 1,
                event_type=event_type,
                payload=payload,
                created_at=dt.datetime.now(dt.timezone.utc),
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return self._to_event(event)

    def list_events(self, run_id: str) -> List[TelemetryEvent]:
        with self.session_factory() as session:
            stmt = (
                select(EventModel)
                .where(EventModel.run_id == run_id)
                .order_by(EventModel.sequence.asc())
            )
            events = session.execute(stmt).scalars().all()
            return [self._to_event(event) for event in events]

    def record_metric(
        self,
        run_id: str,
        *,
        name: str,
        value: float,
        unit: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.session_factory() as session:
            metric = MetricModel(
                run_id=run_id,
                name=name,
                value=value,
                unit=unit,
                meta=metadata,
                recorded_at=dt.datetime.now(dt.timezone.utc),
            )
            session.add(metric)
            session.commit()

    def record_error(
        self,
        run_id: str,
        *,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.session_factory() as session:
            error = ErrorModel(
                run_id=run_id,
                message=message,
                details=details,
                created_at=dt.datetime.now(dt.timezone.utc),
            )
            session.add(error)
            session.commit()

    def record_artifact(
        self,
        run_id: str,
        *,
        name: str,
        uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.session_factory() as session:
            artifact = ArtifactModel(
                run_id=run_id,
                name=name,
                uri=uri,
                meta=metadata,
                created_at=dt.datetime.now(dt.timezone.utc),
            )
            session.add(artifact)
            session.commit()

    def _to_run(self, model: RunModel) -> TelemetryRun:
        return TelemetryRun(
            id=model.id,
            run_type=model.run_type,
            status=model.status,
            input_hash=model.input_hash,
            summary=model.summary,
            created_at=model.created_at,
            updated_at=model.updated_at,
            completed_at=model.completed_at,
        )

    def _to_event(self, model: EventModel) -> TelemetryEvent:
        return TelemetryEvent(
            id=model.id,
            run_id=model.run_id,
            sequence=model.sequence,
            event_type=model.event_type,
            payload=model.payload,
            created_at=model.created_at,
        )


__all__ = [
    "TelemetryRepo",
    "TelemetryRun",
    "TelemetryEvent",
    "Base",
]
