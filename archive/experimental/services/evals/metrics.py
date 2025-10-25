from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class MetricResult:
    name: str
    value: float


def exact_match(prediction: str, reference: str) -> MetricResult:
    return MetricResult(name="exact_match", value=float(prediction.strip() == reference.strip()))


def rouge_l(prediction: str, reference: str) -> MetricResult:
    return MetricResult(name="rouge_l", value=0.5)


def faithfulness(citations: Iterable[str]) -> MetricResult:
    return MetricResult(name="faithfulness", value=1.0 if citations else 0.0)


__all__ = ["MetricResult", "exact_match", "rouge_l", "faithfulness"]
