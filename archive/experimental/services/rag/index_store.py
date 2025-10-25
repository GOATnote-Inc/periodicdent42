from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from core import stable_hash
from services.rag.embed import Embedder
from services.rag.types import CorpusChunk, RetrievalHit


class RagIndex:
    def __init__(
        self,
        *,
        path: Path,
        ttl_seconds: int,
        expected_hash: str | None,
        embedder: Embedder,
    ) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.expected_hash = expected_hash
        self.embedder = embedder
        self.meta: dict[str, object] = {}
        self._chunks: list[CorpusChunk] = []
        self._vectors: list[list[float]] = []
        self._load_from_disk()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        ttl_seconds: int = 86_400,
        expected_hash: str | None = None,
        embedder: Embedder | None = None,
    ) -> "RagIndex":
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        return cls(
            path=target,
            ttl_seconds=ttl_seconds,
            expected_hash=expected_hash,
            embedder=embedder or Embedder(),
        )

    def add(self, docs: Sequence[CorpusChunk]) -> None:
        if not docs:
            return
        embeddings = self.embedder.embed_documents(doc.content for doc in docs)
        for doc, embedding in zip(docs, embeddings):
            self._chunks.append(doc)
            self._vectors.append(embedding.vector)
        self.meta = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "doc_count": len(self._chunks),
            "content_hash": self._compute_hash(self._chunks),
        }
        self._persist()

    def query(self, query: str, k: int = 10) -> List[RetrievalHit]:
        if not self._chunks:
            return []
        query_embedding = self.embedder.embed_query(query).vector
        scored = []
        for chunk, vector in zip(self._chunks, self._vectors):
            score = float(sum(q * v for q, v in zip(query_embedding, vector)))
            scored.append(RetrievalHit(chunk=chunk, score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def _persist(self) -> None:
        meta_path = self.path / "index_meta.json"
        vectors_path = self.path / "index.json"
        with meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(self.meta, meta_file, indent=2, sort_keys=True)
        payload = [
            {
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "content": chunk.content,
                "vector": vector,
            }
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        with vectors_path.open("w", encoding="utf-8") as vector_file:
            json.dump(payload, vector_file)

    def _load_from_disk(self) -> None:
        meta_path = self.path / "index_meta.json"
        vectors_path = self.path / "index.json"
        if not meta_path.exists() or not vectors_path.exists():
            self._chunks = []
            self._vectors = []
            self.meta = {}
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            payload = json.loads(vectors_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._chunks = []
            self._vectors = []
            self.meta = {}
            return
        if self._is_stale(meta):
            self._chunks = []
            self._vectors = []
            self.meta = {}
            return
        self.meta = meta
        chunks: list[CorpusChunk] = []
        vectors: list[list[float]] = []
        for item in payload:
            chunks.append(
                CorpusChunk(doc_id=item["doc_id"], content=item["content"], section=item["section"])
            )
            vectors.append(item["vector"])
        self._chunks = chunks
        self._vectors = vectors

    def _is_stale(self, meta: dict[str, object]) -> bool:
        created_at_raw = meta.get("created_at")
        if created_at_raw:
            try:
                created_at = dt.datetime.fromisoformat(str(created_at_raw))
                age = dt.datetime.now(dt.timezone.utc) - created_at
                if age.total_seconds() > self.ttl_seconds:
                    return True
            except ValueError:
                return True
        expected_hash = self.expected_hash
        if expected_hash and meta.get("content_hash") != expected_hash:
            return True
        return False

    def _compute_hash(self, chunks: Iterable[CorpusChunk]) -> str:
        values = [f"{chunk.doc_id}:{chunk.section}:{chunk.content}" for chunk in chunks]
        return stable_hash(values)

    def metadata(self) -> dict[str, object]:
        return dict(self.meta)


__all__ = ["RagIndex"]
