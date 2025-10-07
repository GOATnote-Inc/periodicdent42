from __future__ import annotations

import json
import os
from pathlib import Path

from core import stable_hash
from services.rag.index_store import RagIndex
from services.rag.types import CorpusChunk, RetrievalHit


class CorpusIndex:
    """RAG index facade backed by a persistent vector cache."""

    def __init__(self, rag_index: RagIndex):
        self._rag_index = rag_index

    def top_k(self, query: str, k: int = 5) -> list[RetrievalHit]:
        return self._rag_index.query(query, k=k)

    @property
    def meta(self) -> dict[str, object]:
        return self._rag_index.metadata()

    @property
    def doc_count(self) -> int:
        meta = self.meta
        if not meta:
            return 0
        count = meta.get("doc_count", 0)
        return int(count) if isinstance(count, (int, float)) else 0


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


def load_eval_dataset(path: Path) -> list[dict]:
    entries: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def ingest(
    corpus_dir: Path | None = None,
    *,
    cache_dir: Path | None = None,
    ttl_seconds: int = 86_400,
) -> CorpusIndex:
    target_dir = corpus_dir or Path("datasets/synthetic/corpus")
    documents = load_corpus_chunks(target_dir)
    content_hash = stable_hash(f"{doc.doc_id}:{doc.section}:{doc.content}" for doc in documents)
    cache_root = Path(os.getenv("RAG_CACHE_DIR", str(cache_dir or Path(".cache/rag"))))
    ttl = int(os.getenv("RAG_CACHE_TTL_SECONDS", ttl_seconds))
    rag_index = RagIndex.open(cache_root, ttl_seconds=ttl, expected_hash=content_hash)
    if not rag_index.metadata() or rag_index.metadata().get("content_hash") != content_hash:
        rag_index.add(documents)
    return CorpusIndex(rag_index)


__all__ = ["ingest", "CorpusIndex", "CorpusChunk", "RetrievalHit", "load_eval_dataset"]


if __name__ == "__main__":
    index = ingest()
    print(f"Loaded {index.doc_count} chunks from synthetic corpus.")
