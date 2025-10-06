from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from services.telemetry.telemetry_repo import TelemetryRepo


@pytest.fixture
def telemetry_db(tmp_path: Path) -> str:
    """Create temporary test database with Alembic migrations applied."""
    db_path = tmp_path / "telemetry.sqlite"
    url = f"sqlite:///{db_path}"
    
    # Resolve Alembic config path relative to repository root
    # This works in both local development and CI environments
    repo_root = Path(__file__).resolve().parent.parent
    alembic_ini = repo_root / "infra" / "db" / "alembic.ini"
    migrations_dir = repo_root / "infra" / "db" / "migrations"
    
    cfg = Config(str(alembic_ini))
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    return url


def test_create_run_and_events(telemetry_db: str) -> None:
    repo = TelemetryRepo(telemetry_db)
    run = repo.create_run(run_type="chat", status="running", summary={"phase": "start"})
    repo.append_event(run.id, event_type="router", payload={"arm": "balanced"})
    repo.record_metric(run.id, name="latency_ms", value=42.0)
    repo.record_artifact(run.id, name="rag_index_meta", metadata={"doc_count": 5})

    updated = repo.update_run_status(run.id, status="completed", summary={"answer": "ok"})
    assert updated.status == "completed"
    assert updated.summary["answer"] == "ok"

    events = repo.list_events(run.id)
    assert events[0].payload["arm"] == "balanced"

    runs = repo.list_runs(limit=5)
    assert runs[0].id == run.id


def test_soft_delete_excludes_runs(telemetry_db: str) -> None:
    repo = TelemetryRepo(telemetry_db)
    run = repo.create_run(run_type="chat", status="running")
    repo.soft_delete(run.id)
    assert repo.list_runs(limit=5) == []


def test_pagination(telemetry_db: str) -> None:
    repo = TelemetryRepo(telemetry_db)
    for idx in range(5):
        repo.create_run(run_type="chat", status=f"state-{idx}")
    runs = repo.list_runs(limit=3)
    assert len(runs) == 3
    assert runs[0].status.startswith("state")
