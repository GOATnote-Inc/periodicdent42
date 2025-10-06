from __future__ import annotations

from dataclasses import dataclass
from typing import List

from services.llm.client import ChatCompletion, LLMClient, Message
from services.llm.guardrails import run_guardrails
from services.llm.router import RouterDecision
from services.rag.index import CorpusIndex, RetrievalHit, ingest
from services.rag.models import ChatRequest, ChatResponse, Citation, VectorStats


@dataclass
class PipelineResult:
    response: ChatResponse
    completion: ChatCompletion
    hits: List[RetrievalHit]


class ChatPipeline:
    def __init__(self, llm_client: LLMClient, index: CorpusIndex):
        self.llm_client = llm_client
        self.index = index

    @classmethod
    def default(cls) -> "ChatPipeline":
        index = ingest()
        llm_client = LLMClient.fake()
        return cls(llm_client=llm_client, index=index)

    def run(self, request: ChatRequest, router_decision: RouterDecision | None) -> PipelineResult:
        hits = self.index.top_k(request.query, k=5)
        messages = self._build_prompt(request.query, hits)
        completion = self.llm_client.complete(messages)
        citations = [
            Citation(doc_id=hit.chunk.doc_id, section=hit.chunk.section, text=hit.chunk.content)
            for hit in hits
        ]
        vector_stats = self._compute_vector_stats(hits)
        router_payload = {
            "arm": router_decision.arm if router_decision else "balanced",
            "policy": router_decision.policy if router_decision else "default",
        }
        guardrail_results = run_guardrails(request.query, citations)
        guardrail_status = [
            {"rule": result.name, "status": "pass" if result.passed else "fail", "details": result.details}
            for result in guardrail_results
        ]
        response = ChatResponse(
            answer=completion.text,
            citations=citations,
            router=router_payload,
            guardrails=guardrail_status,
            vector_stats=vector_stats,
        )
        return PipelineResult(response=response, completion=completion, hits=hits)

    @property
    def index_meta(self) -> dict[str, object]:
        meta_candidate = getattr(self.index, "meta", {})
        if callable(meta_candidate):
            meta_candidate = meta_candidate()
        return meta_candidate if isinstance(meta_candidate, dict) else {}

    def _build_prompt(self, query: str, hits: List[RetrievalHit]) -> List[Message]:
        context = "\n".join(f"Source {idx+1}: {hit.chunk.content}" for idx, hit in enumerate(hits))
        system = Message(role="system", content="Answer with citations in brackets like [1].")
        user = Message(role="user", content=f"Question: {query}\nContext:\n{context}")
        return [system, user]

    def _compute_vector_stats(self, hits: List[RetrievalHit]) -> VectorStats:
        if not hits:
            return VectorStats(avg_similarity=0.0, retrieved=0, ann_probe_ms=0.0)
        average_score = sum(hit.score for hit in hits) / len(hits)
        return VectorStats(avg_similarity=average_score, retrieved=len(hits), ann_probe_ms=1.2)


__all__ = [
    "ChatPipeline",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "VectorStats",
    "PipelineResult",
]
