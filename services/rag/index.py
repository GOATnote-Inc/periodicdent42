from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from services.rag.embed import Embedder, EmbeddingResult


@dataclass
class CorpusChunk:
    doc_id: str
    content: str
    section: str


@dataclass
class RetrievalHit:
    chunk: CorpusChunk
    score: float


class CorpusIndex:
    """In-memory hybrid index placeholder."""

    def __init__(self, chunks: list[CorpusChunk], embeddings: list[EmbeddingResult]):
        self.chunks = chunks
        self.embeddings = embeddings

    def top_k(self, query: str, k: int = 5) -> List[RetrievalHit]:
        # Simplified scoring based on keyword overlap.
        query_terms = set(query.lower().split())
        hits: list[RetrievalHit] = []
        for chunk in self.chunks:
            score = sum(1 for term in query_terms if term in chunk.content.lower())
            if score:
                hits.append(RetrievalHit(chunk=chunk, score=float(score)))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]


def load_corpus_chunks(corpus_dir: Path) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text()
        sections = text.split("## ")
        for section_text in sections:
            if not section_text.strip():
                continue
            lines = section_text.splitlines()
            title = lines[0].strip()
            body = "\n".join(lines[1:])
            chunks.append(
                CorpusChunk(doc_id=path.stem, content=body.strip(), section=title or "Body")
            )
    return chunks


def build_in_memory_index(corpus_dir: Path) -> CorpusIndex:
    embedder = Embedder()
    chunks = load_corpus_chunks(corpus_dir)
    embeddings = embedder.embed_documents(chunk.content for chunk in chunks)
    return CorpusIndex(chunks=chunks, embeddings=embeddings)


def load_eval_dataset(path: Path) -> list[dict]:
    entries: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def ingest(corpus_dir: Path | None = None) -> CorpusIndex:
    target_dir = corpus_dir or Path("datasets/synthetic/corpus")
    return build_in_memory_index(target_dir)


__all__ = ["ingest", "CorpusIndex", "CorpusChunk", "RetrievalHit", "load_eval_dataset"]


if __name__ == "__main__":
    index = ingest()
    print(f"Loaded {len(index.chunks)} chunks from synthetic corpus.")
