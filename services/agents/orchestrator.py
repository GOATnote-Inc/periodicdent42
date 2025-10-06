from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from services.rag.models import ChatRequest, ChatResponse
from services.rag.pipeline import ChatPipeline, PipelineResult
from services.router.llm_router import LLMRouter
from services.telemetry.store import RouterLog, TelemetryStore


@dataclass
class ToolCall:
    name: str
    input: str
    output: str


@dataclass
class AgentTrace:
    steps: List[ToolCall]


class Orchestrator:
    def __init__(
        self,
        pipeline: ChatPipeline,
        telemetry_store: TelemetryStore,
        router: LLMRouter,
    ) -> None:
        self.pipeline = pipeline
        self.telemetry_store = telemetry_store
        self.router = router

    def run(self, request: ChatRequest) -> tuple[ChatResponse, AgentTrace]:
        router_trace = self.router.route(request)
        start = time.perf_counter()
        result: PipelineResult = self.pipeline.run(request, router_decision=router_trace.decision)
        latency_ms = (time.perf_counter() - start) * 1000
        result_id = f"{result.completion.provider}-{router_trace.input_hash[:8]}"
        router_log: RouterLog = router_trace.finalize(
            latency_ms=latency_ms,
            tokens_in=result.completion.tokens_in,
            tokens_out=result.completion.tokens_out,
            provider=result.completion.provider,
            result_id=result_id,
        )
        self.router.emit(router_log)

        run_record = self.telemetry_store.log_chat_run(
            request=request,
            response=result.response,
            router_log=router_log,
            latency_ms=latency_ms,
        )
        meta_payload: dict[str, object]
        meta_payload = self.pipeline.index_meta
        self.telemetry_store.record_artifact(
            run_record.id,
            name="rag_index_meta",
            metadata=meta_payload,
        )

        trace = AgentTrace(
            steps=[
                ToolCall(name="router", input=request.query, output=router_log.reason),
                ToolCall(name="rag_query", input=request.query, output=f"retrieved {len(result.hits)} chunks"),
            ]
        )
        return result.response, trace


__all__ = ["Orchestrator", "AgentTrace", "ToolCall"]
