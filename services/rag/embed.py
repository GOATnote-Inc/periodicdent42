from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class EmbeddingResult:
    text: str
    vector: list[float]


class EmbeddingBackend:
    """Simple registry for pluggable embedding providers."""

    OPENAI = "openai"
    VERTEX = "vertex"
    LOCAL = "local"
    FAKE = "fake"


@dataclass
class EmbeddingConfig:
    backend: str = EmbeddingBackend.FAKE
    dimension: int = 128
    seed: int = 13


class Embedder:
    """Utility class that simulates deterministic embedding generation."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.dimension = self.config.dimension

    def _vector_for_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed_value = int.from_bytes(digest[:8], "big") ^ self.config.seed
        random.seed(seed_value)
        vector = [random.uniform(-1.0, 1.0) for _ in range(self.dimension)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: Iterable[str]) -> List[EmbeddingResult]:
        return [EmbeddingResult(text=text, vector=self._vector_for_text(text)) for text in texts]

    def embed_query(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(text=text, vector=self._vector_for_text(text))


__all__ = ["Embedder", "EmbeddingConfig", "EmbeddingResult", "EmbeddingBackend"]
