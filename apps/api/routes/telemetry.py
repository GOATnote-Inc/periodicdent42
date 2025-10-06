from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query

from services.telemetry.store import TelemetryStore
from services.telemetry.telemetry_repo import TelemetryEvent, TelemetryRun

router = APIRouter()
_store: TelemetryStore | None = None


def configure(store: TelemetryStore) -> None:
    global _store
    _store = store


def get_store() -> TelemetryStore:
    if _store is None:  # pragma: no cover - configuration happens at import time
        raise RuntimeError("telemetry store not configured")
    return _store


def serialize_run(run: TelemetryRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "run_type": run.run_type,
        "status": run.status,
        "input_hash": run.input_hash,
        "summary": run.summary or {},
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


def serialize_event(event: TelemetryEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "payload": event.payload or {},
        "created_at": event.created_at.isoformat(),
    }


@router.get("/runs")
def list_runs(
    limit: int = Query(50, ge=1, le=200),
    status: str | None = Query(default=None),
    store: TelemetryStore = Depends(get_store),
) -> dict[str, List[dict[str, Any]]]:
    runs = store.list_runs(limit=limit, status=status)
    return {"runs": [serialize_run(run) for run in runs]}


@router.get("/runs/{run_id}")
def get_run(run_id: str, store: TelemetryStore = Depends(get_store)) -> dict[str, Any]:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_not_found")
    return serialize_run(run)


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str, store: TelemetryStore = Depends(get_store)) -> dict[str, List[dict[str, Any]]]:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_not_found")
    events = store.list_events(run_id)
    return {"events": [serialize_event(event) for event in events]}


__all__ = ["router", "configure"]
