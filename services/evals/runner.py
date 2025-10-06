from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from services.evals.metrics import MetricResult, exact_match, faithfulness, rouge_l
from services.rag.index import load_eval_dataset
from services.rag.models import ChatRequest
from services.rag.pipeline import ChatPipeline


@dataclass
class EvalExampleResult:
    example_id: str
    metrics: List[MetricResult]


@dataclass
class EvalRun:
    run_id: str
    results: List[EvalExampleResult]


def run_offline_eval(dataset_path: Path | None = None) -> EvalRun:
    path = dataset_path or Path("datasets/synthetic/eval.jsonl")
    dataset = load_eval_dataset(path)
    pipeline = ChatPipeline.default()
    results: list[EvalExampleResult] = []
    for example in dataset[:5]:  # limit for stub
        request = ChatRequest(query=example["question"])
        result = pipeline.run(request, router_decision=None)
        response = result.response
        metrics = [
            exact_match(response.answer, example["answer"]),
            rouge_l(response.answer, example["answer"]),
            faithfulness([citation.doc_id for citation in response.citations]),
        ]
        results.append(EvalExampleResult(example_id=example["id"], metrics=metrics))
    return EvalRun(run_id="offline-stub", results=results)


__all__ = ["run_offline_eval", "EvalRun", "EvalExampleResult"]


if __name__ == "__main__":
    run = run_offline_eval()
    print(f"Completed eval run {run.run_id} with {len(run.results)} examples.")
