from __future__ import annotations

import json
import logging
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

try:  # pragma: no cover - optional dependency shim
    from prometheus_client import Counter  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _CounterHandle:
        def __init__(self, storage: Dict[Tuple[Tuple[str, str], ...], float], key: Tuple[Tuple[str, str], ...]):
            self._storage = storage
            self._key = key

        def inc(self, amount: float = 1.0) -> None:
            self._storage[self._key] = self._storage.get(self._key, 0.0) + amount

    class Counter:  # type: ignore[override]
        def __init__(self, name: str, doc: str, labelnames: list[str] | tuple[str, ...]):
            self._labelnames = tuple(labelnames)
            self._storage: Dict[Tuple[Tuple[str, str], ...], float] = {}

        def labels(self, **labels: str) -> _CounterHandle:
            key = tuple((label, str(labels.get(label, ""))) for label in self._labelnames)
            return _CounterHandle(self._storage, key)

        def collect(self):  # mimics prometheus_client metric
            sample_cls = type("Sample", (), {})
            metric_cls = type("Metric", (), {})
            metric = metric_cls()
            metric.samples = []
            for key, value in self._storage.items():
                sample = sample_cls()
                sample.labels = dict(key)
                sample.value = value
                metric.samples.append(sample)
            return [metric]

from core import stable_hash
from services.rag.models import ChatRequest

ROUTER_DECISIONS = Counter(
    "router_decisions_total",
    "Count of router arm selections",
    ["arm", "reason"],
)


@dataclass
class RouterConfig:
    latency_budget_ms: float = 800.0
    max_context_tokens: int = 280
    uncertainty_threshold: float = 0.55

    @classmethod
    def from_config(cls, path: Path | None = None) -> "RouterConfig":
        config_path = path or Path(os.getenv("SERVICE_CONFIG", "configs/service.yaml"))
        if not config_path.exists():
            return cls()
        data = yaml.safe_load(config_path.read_text()) or {}
        router_cfg = data.get("router", {})
        return cls(
            latency_budget_ms=float(router_cfg.get("latency_budget_ms", cls.latency_budget_ms)),
            max_context_tokens=int(router_cfg.get("max_context_tokens", cls.max_context_tokens)),
            uncertainty_threshold=float(router_cfg.get("uncertainty_threshold", cls.uncertainty_threshold)),
        )

    def apply_env(self) -> "RouterConfig":
        latency = os.getenv("ROUTER_LATENCY_BUDGET_MS")
        context = os.getenv("ROUTER_MAX_CONTEXT_TOKENS")
        uncertainty = os.getenv("ROUTER_UNCERTAINTY_THRESHOLD")
        if latency:
            self.latency_budget_ms = float(latency)
        if context:
            self.max_context_tokens = int(context)
        if uncertainty:
            self.uncertainty_threshold = float(uncertainty)
        return self


@dataclass
class RouterDecision:
    arm: str
    policy: str


@dataclass
class RouterTrace:
    input_hash: str
    decision: RouterDecision
    reason: str
    context_tokens: int
    estimated_latency_ms: float
    uncertainty: float

    def finalize(
        self,
        *,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        provider: str,
        result_id: str,
    ) -> "RouterLog":
        return RouterLog(
            input_hash=self.input_hash,
            arm=self.decision.arm,
            policy=self.decision.policy,
            reason=self.reason,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            provider=provider,
            result_id=result_id,
            context_tokens=self.context_tokens,
            estimated_latency_ms=self.estimated_latency_ms,
            uncertainty=self.uncertainty,
        )


@dataclass
class RouterLog:
    input_hash: str
    arm: str
    policy: str
    reason: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    provider: str
    result_id: str
    context_tokens: int
    estimated_latency_ms: float
    uncertainty: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "arm": self.arm,
            "policy": self.policy,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "provider": self.provider,
            "result_id": self.result_id,
            "context_tokens": self.context_tokens,
            "estimated_latency_ms": self.estimated_latency_ms,
            "uncertainty": self.uncertainty,
        }


class LLMRouter:
    def __init__(self, config: RouterConfig | None = None, logger: logging.Logger | None = None) -> None:
        self.config = (config or RouterConfig.from_config()).apply_env()
        self.logger = logger or logging.getLogger(__name__)

    def route(self, request: ChatRequest) -> RouterTrace:
        input_hash = stable_hash([request.query])
        context_tokens = len(request.query.split())
        estimated_latency = context_tokens * 2.5
        uncertainty = self._estimate_uncertainty(request.query)

        if request.arm:
            decision = RouterDecision(arm=request.arm, policy="user_override")
            reason = "user_override"
        elif context_tokens > self.config.max_context_tokens:
            decision = RouterDecision(arm="high-accuracy", policy="context-budget")
            reason = "context_budget_exceeded"
        elif estimated_latency > self.config.latency_budget_ms:
            decision = RouterDecision(arm="flash", policy="latency-budget")
            reason = "latency_budget_exceeded"
        elif uncertainty >= self.config.uncertainty_threshold:
            decision = RouterDecision(arm="high-accuracy", policy="uncertainty")
            reason = "uncertainty_high"
        else:
            decision = RouterDecision(arm="balanced", policy="default")
            reason = "default"

        trace = RouterTrace(
            input_hash=input_hash,
            decision=decision,
            reason=reason,
            context_tokens=context_tokens,
            estimated_latency_ms=estimated_latency,
            uncertainty=uncertainty,
        )
        return trace

    def emit(self, log: RouterLog) -> None:
        ROUTER_DECISIONS.labels(arm=log.arm, reason=log.reason).inc()
        self.logger.info("router_decision", extra={"payload": json.dumps(log.as_dict())})

    def _estimate_uncertainty(self, text: str) -> float:
        if not text:
            return 0.0
        question_marks = text.count("?")
        speculative_terms = sum(1 for term in ["maybe", "perhaps", "could"] if term in text.lower())
        score = min(1.0, 0.2 * question_marks + 0.15 * speculative_terms)
        return score


__all__ = [
    "LLMRouter",
    "RouterConfig",
    "RouterDecision",
    "RouterLog",
    "RouterTrace",
]
