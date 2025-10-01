from __future__ import annotations

from typing import Iterable, List

from .base import BaseAdapter
from ..models.schemas import WorkflowEvent


class HTTPAdapter(BaseAdapter):
    """Adapter placeholder for webhook ingested events."""

    def __init__(self, buffer: List[WorkflowEvent] | None = None):
        self.buffer = buffer or []

    def add_event(self, event: WorkflowEvent) -> None:
        self.buffer.append(event)

    def stream(self) -> Iterable[WorkflowEvent]:
        while self.buffer:
            yield self.buffer.pop(0)
