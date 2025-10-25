from __future__ import annotations

from fastapi import FastAPI, HTTPException

from prometheus_client import Counter

from services.agents.orchestrator import Orchestrator
from services.rag.models import ChatRequest, ChatResponse
from services.rag.pipeline import ChatPipeline
from services.router.llm_router import LLMRouter
from services.telemetry.store import TelemetryStore
from apps.api.routes.telemetry import configure as configure_telemetry_router, router as telemetry_router


app = FastAPI(title="RAG Mastery Demo API", version="0.1.0")
telemetry_store = TelemetryStore.from_env()
pipeline = ChatPipeline.default()
router = LLMRouter()
orchestrator = Orchestrator(pipeline=pipeline, telemetry_store=telemetry_store, router=router)
configure_telemetry_router(telemetry_store)

API_REQUESTS = Counter("api_requests_total", "API requests", ["route"])
API_ERRORS = Counter("api_errors_total", "API errors", ["route"])

app.include_router(telemetry_router, prefix="/api/telemetry", tags=["telemetry"])


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    API_REQUESTS.labels(route="chat").inc()
    try:
        response, _trace = orchestrator.run(request)
    except Exception as exc:  # pragma: no cover - FastAPI will handle logging
        API_ERRORS.labels(route="chat").inc()
        raise HTTPException(status_code=500, detail="chat_failed") from exc
    return response


@app.post("/api/evals/run")
def run_evals() -> dict[str, str]:
    return {"status": "scheduled", "run_id": "offline-local"}


@app.get("/api/evals/{run_id}")
def get_eval(run_id: str) -> dict[str, str]:
    return {"run_id": run_id, "status": "pending"}


@app.get("/api/errors")
def get_errors() -> dict[str, list[dict[str, str]]]:
    return {"clusters": []}
