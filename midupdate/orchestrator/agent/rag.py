from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Evidence:
    source_id: str
    span: str
    confidence: float


class LocalRAGIndex:
    """Tiny local retrieval index backed by processed dataset."""

    def __init__(self, data_dir: Path) -> None:
        self.documents = self._load_documents(data_dir)

    @staticmethod
    def _load_documents(data_dir: Path) -> List[dict]:
        docs: List[dict] = []
        for split in ("train", "val", "test"):
            path = data_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    payload["_source_id"] = f"{split}:{idx}"
                    docs.append(payload)
        if not docs:
            docs.append(
                {
                    "goal": "maximize critical temperature",
                    "plan_text": "anneal and measure",
                    "rationale_text": "increase ordering",
                    "_source_id": "synthetic:0",
                }
            )
        return docs

    def search(self, query: str, top_k: int = 3) -> List[Evidence]:
        results: List[Evidence] = []
        for doc in self.documents[:top_k]:
            snippet = doc.get("rationale_text") or doc.get("plan_text", "")
            results.append(
                Evidence(
                    source_id=doc.get("_source_id", "unknown"),
                    span=snippet[:200],
                    confidence=0.6,
                )
            )
        return results
