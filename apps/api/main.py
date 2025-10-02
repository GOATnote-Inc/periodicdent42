from __future__ import annotations

from fastapi import FastAPI

from services.llm.router import RouterDecision, select_arm
from services.rag.models import ChatRequest, ChatResponse
from services.rag.pipeline import ChatPipeline
from services.telemetry.store import TelemetryStore


app = FastAPI(title="RAG Mastery Demo API", version="0.1.0")
telemetry_store = TelemetryStore.in_memory()
pipeline = ChatPipeline.default(telemetry_store=telemetry_store)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    decision: RouterDecision = select_arm(request)
    response = pipeline.run(request=request, router_decision=decision)
    telemetry_store.log_chat(response)
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
