from __future__ import annotations

from typing import List

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    arm: str | None = None


class Citation(BaseModel):
    doc_id: str
    section: str
    text: str


class VectorStats(BaseModel):
    avg_similarity: float
    retrieved: int
    ann_probe_ms: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    router: dict[str, str]
    guardrails: List[dict[str, str]]
    vector_stats: VectorStats


__all__ = ["ChatRequest", "Citation", "VectorStats", "ChatResponse"]
