from __future__ import annotations

import time
from pathlib import Path

from services.rag.types import CorpusChunk
from services.rag.index_store import RagIndex
from services.rag.embed import Embedder


class RecordingEmbedder(Embedder):
    def __init__(self, delay: float = 0.0):
        super().__init__()
        self.delay = delay
        self.calls = 0

    def embed_documents(self, texts):  # type: ignore[override]
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return super().embed_documents(texts)

    def embed_query(self, text):  # type: ignore[override]
        if self.delay:
            time.sleep(self.delay)
        return super().embed_query(text)


def test_rag_index_persists_vectors(tmp_path: Path) -> None:
    docs = [
        CorpusChunk(doc_id="doc1", section="Intro", content="Quantum materials overview"),
        CorpusChunk(doc_id="doc2", section="Methods", content="High throughput screening techniques"),
    ]
    cache_dir = tmp_path / "cache"
    slow_embedder = RecordingEmbedder(delay=0.02)
    index = RagIndex.open(cache_dir, expected_hash=None, embedder=slow_embedder)

    start = time.perf_counter()
    index.add(docs)
    first_duration = time.perf_counter() - start
    assert slow_embedder.calls == 1
    assert index.query("screening")

    meta = index.metadata()
    fast_embedder = RecordingEmbedder()
    start_fast = time.perf_counter()
    cached_index = RagIndex.open(cache_dir, expected_hash=str(meta.get("content_hash")), embedder=fast_embedder)
    second_duration = time.perf_counter() - start_fast
    assert fast_embedder.calls == 0
    assert cached_index.query("screening")
    assert second_duration < first_duration

    stale_index = RagIndex.open(cache_dir, ttl_seconds=0, expected_hash=str(meta.get("content_hash")), embedder=fast_embedder)
    assert stale_index.metadata() == {}
