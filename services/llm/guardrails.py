from __future__ import annotations

from dataclasses import dataclass
from typing import List

from services.rag.models import Citation


@dataclass
class GuardrailResult:
    name: str
    passed: bool
    details: str


PII_PATTERNS = ["@", "phone", "ssn"]


def run_guardrails(query: str, citations: List[Citation]) -> List[GuardrailResult]:
    flags: list[GuardrailResult] = []
    contains_pii = any(pattern in query.lower() for pattern in PII_PATTERNS)
    flags.append(GuardrailResult(name="pii", passed=not contains_pii, details="basic heuristic"))
    grounded = bool(citations)
    flags.append(GuardrailResult(name="grounding", passed=grounded, details="requires citations"))
    return flags


__all__ = ["run_guardrails", "GuardrailResult"]
