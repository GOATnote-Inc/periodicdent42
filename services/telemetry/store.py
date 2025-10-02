from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from services.rag.models import ChatResponse


@dataclass
class TelemetryRecord:
    answer: str
    arm: str
    policy: str


@dataclass
class TelemetryStore:
    records: List[TelemetryRecord] = field(default_factory=list)

    @classmethod
    def in_memory(cls) -> "TelemetryStore":
        return cls()

    def log_chat(self, response: ChatResponse) -> None:
        self.records.append(
            TelemetryRecord(answer=response.answer, arm=response.router["arm"], policy=response.router["policy"])
        )


__all__ = ["TelemetryStore", "TelemetryRecord"]
