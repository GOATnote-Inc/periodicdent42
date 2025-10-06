from __future__ import annotations

import json

import pytest

from services.rag.models import ChatRequest
from services.router.llm_router import LLMRouter, RouterConfig, ROUTER_DECISIONS


def _get_metric(arm: str, reason: str) -> float:
    for sample in ROUTER_DECISIONS.collect()[0].samples:
        if sample.labels.get("arm") == arm and sample.labels.get("reason") == reason:
            return sample.value
    return 0.0


def test_router_user_override() -> None:
    router = LLMRouter(RouterConfig(latency_budget_ms=1000, max_context_tokens=10, uncertainty_threshold=0.9))
    request = ChatRequest(query="short question", arm="flash")
    trace = router.route(request)
    assert trace.decision.arm == "flash"
    assert trace.reason == "user_override"


def test_router_logging_and_metrics(caplog: pytest.LogCaptureFixture) -> None:
    router = LLMRouter(RouterConfig(latency_budget_ms=10, max_context_tokens=3, uncertainty_threshold=0.2))
    request = ChatRequest(query="maybe this works?", arm=None)
    trace = router.route(request)
    log = trace.finalize(
        latency_ms=120.0,
        tokens_in=50,
        tokens_out=25,
        provider="fake",
        result_id="result-1",
    )

    before = _get_metric(log.arm, log.reason)
    with caplog.at_level("INFO"):
        router.emit(log)
    after = _get_metric(log.arm, log.reason)
    assert after == pytest.approx(before + 1.0)

    record = caplog.records[-1]
    assert record.message == "router_decision"
    payload = json.loads(record.payload)
    assert payload["reason"] == log.reason
    assert payload["tokens_in"] == 50
