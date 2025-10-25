from __future__ import annotations

from typing import Protocol

from services.router.llm_router import LLMRouter, RouterDecision


class SupportsRouterContext(Protocol):
    query: str
    arm: str | None


_ROUTER = LLMRouter()


def select_arm(request: SupportsRouterContext) -> RouterDecision:
    trace = _ROUTER.route(request)
    return trace.decision


__all__ = ["RouterDecision", "select_arm"]
