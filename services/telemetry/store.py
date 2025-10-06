from __future__ import annotations

import os
from typing import Any, Iterable, List

import datetime as dt

from services.rag.models import ChatRequest, ChatResponse
from services.router.llm_router import RouterLog
from services.telemetry.telemetry_repo import TelemetryRepo, TelemetryRun


class TelemetryStore:
    def __init__(self, repo: TelemetryRepo):
        self.repo = repo

    @classmethod
    def from_env(cls) -> "TelemetryStore":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            user = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            name = os.getenv("DB_NAME")
            if user and password and name:
                database_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
        if not database_url:
            database_url = "sqlite:///./telemetry.db"
        return cls(TelemetryRepo(database_url))

    def start_run(
        self,
        *,
        run_type: str,
        input_hash: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> TelemetryRun:
        return self.repo.create_run(run_type=run_type, status="running", input_hash=input_hash, summary=summary)

    def complete_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        summary: dict[str, Any] | None = None,
    ) -> TelemetryRun:
        completed = dt.datetime.now(dt.timezone.utc) if status == "completed" else None
        return self.repo.update_run_status(
            run_id,
            status=status,
            summary=summary,
            completed_at=completed,
        )

    def append_router_event(self, run_id: str, router_log: RouterLog) -> None:
        self.repo.append_event(run_id, event_type="router_decision", payload=router_log.as_dict())

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.repo.append_event(run_id, event_type=event_type, payload=payload)

    def record_metrics(self, run_id: str, metrics: Iterable[tuple[str, float, dict[str, Any] | None]]):
        for name, value, metadata in metrics:
            self.repo.record_metric(run_id, name=name, value=value, metadata=metadata)

    def record_artifact(
        self,
        run_id: str,
        *,
        name: str,
        uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.repo.record_artifact(run_id, name=name, uri=uri, metadata=metadata)

    def record_error(self, run_id: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.repo.record_error(run_id, message=message, details=details)

    def list_runs(self, *, limit: int = 50, status: str | None = None) -> List[TelemetryRun]:
        return self.repo.list_runs(limit=limit, status=status)

    def get_run(self, run_id: str) -> TelemetryRun | None:
        return self.repo.get_run(run_id)

    def list_events(self, run_id: str):
        return self.repo.list_events(run_id)

    def log_chat_run(
        self,
        *,
        request: ChatRequest,
        response: ChatResponse,
        router_log: RouterLog,
        latency_ms: float,
    ) -> TelemetryRun:
        run = self.start_run(
            run_type="chat",
            input_hash=router_log.input_hash,
            summary={"question": request.query},
        )
        self.append_router_event(run.id, router_log)
        self.record_metrics(
            run.id,
            [
                ("latency_ms", latency_ms, {"phase": "chat"}),
                ("tokens_in", float(router_log.tokens_in), {"provider": router_log.provider}),
                ("tokens_out", float(router_log.tokens_out), {"provider": router_log.provider}),
                (
                    "context_tokens",
                    float(router_log.context_tokens),
                    {"estimated_latency_ms": router_log.estimated_latency_ms},
                ),
            ],
        )
        self.repo.record_metric(
            run.id,
            name="citations",
            value=float(len(response.citations)),
            metadata={"provider": router_log.provider},
        )
        self.repo.record_metric(
            run.id,
            name="avg_similarity",
            value=response.vector_stats.avg_similarity,
            metadata={"retrieved": response.vector_stats.retrieved},
        )
        summary = {
            "answer": response.answer,
            "router": router_log.as_dict(),
            "vector_stats": {
                "avg_similarity": response.vector_stats.avg_similarity,
                "retrieved": response.vector_stats.retrieved,
            },
        }
        return self.complete_run(run.id, summary=summary)


__all__ = ["TelemetryStore", "RouterLog"]
