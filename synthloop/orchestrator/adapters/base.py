from __future__ import annotations

import threading
from datetime import timedelta
from typing import Protocol

from ..models.plan import Step, SynthesisPlan


class Rig(Protocol):
    def preflight(self, plan: SynthesisPlan) -> dict:
        ...

    def execute_step(self, step: Step, cancel: threading.Event) -> dict:
        ...

    def estimate_duration(self, plan: SynthesisPlan) -> timedelta:
        ...

