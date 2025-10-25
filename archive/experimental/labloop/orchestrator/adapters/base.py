from __future__ import annotations

import threading
from datetime import timedelta
from typing import Protocol

from ..models.schemas import ExperimentPlan, Task


class PreflightReport:
    def __init__(self, ok: bool, message: str = "") -> None:
        self.ok = ok
        self.message = message


class Measurement:
    def __init__(self, task_id: str, data: dict) -> None:
        self.task_id = task_id
        self.data = data


class Instrument(Protocol):
    def preflight(self, plan: ExperimentPlan) -> PreflightReport:
        ...

    def execute(self, task: Task, cancel_token: threading.Event) -> Measurement:
        ...

    def estimate_duration(self, task: Task) -> timedelta:
        ...
