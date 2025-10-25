from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..models.schemas import WorkflowEvent


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, events: Iterable[WorkflowEvent]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            for event in events:
                record = event.dict()
                record["ts"] = event.ts.isoformat()
                fh.write(json.dumps(record))
                fh.write("\n")

    def read_all(self) -> list[WorkflowEvent]:
        events: list[WorkflowEvent] = []
        if not self.path.exists():
            return events
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                payload = json.loads(line)
                payload["ts"] = payload["ts"]
                events.append(WorkflowEvent.parse_obj(payload))
        return events
