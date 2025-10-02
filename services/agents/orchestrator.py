from __future__ import annotations

from dataclasses import dataclass
from typing import List

from services.rag.models import ChatRequest, ChatResponse
from services.rag.pipeline import ChatPipeline


@dataclass
class ToolCall:
    name: str
    input: str
    output: str


@dataclass
class AgentTrace:
    steps: List[ToolCall]


class Orchestrator:
    def __init__(self, pipeline: ChatPipeline) -> None:
        self.pipeline = pipeline

    def run(self, request: ChatRequest) -> tuple[ChatResponse, AgentTrace]:
        tool_call = ToolCall(name="rag_query", input=request.query, output="retrieved synthetic context")
        response = self.pipeline.run(request, router_decision=None)  # type: ignore[arg-type]
        trace = AgentTrace(steps=[tool_call])
        return response, trace


__all__ = ["Orchestrator", "AgentTrace", "ToolCall"]
