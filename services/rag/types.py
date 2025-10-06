from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorpusChunk:
    doc_id: str
    content: str
    section: str


@dataclass
class RetrievalHit:
    chunk: "CorpusChunk"
    score: float


__all__ = ["CorpusChunk", "RetrievalHit"]
