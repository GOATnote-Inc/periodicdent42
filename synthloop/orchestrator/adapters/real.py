from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta

from ..models.plan import Step, SynthesisPlan
from ..safety import interlocks


class RealRig:
    def __init__(self):
        required_env = ["MAX_T_C", "MAX_RAMP_C_PER_MIN", "MAX_RPM", "MAX_BATCH_G"]
        missing = [name for name in required_env if name not in os.environ]
        if missing:
            raise RuntimeError(f"Missing safety limits: {missing}")
        self.interlock_state = {
            "EnclosureClosed": False,
            "EStopNotEngaged": False,
            "VentilationOn": False,
            "ScaleHealthy": False,
            "PowerOK": False,
        }

    def preflight(self, plan: SynthesisPlan) -> dict:
        statuses = interlocks.verify(plan, self.interlock_state)
        failed = [s for s in statuses if not s.ok]
        if failed:
            raise RuntimeError(f"Interlocks not satisfied: {[s.name for s in failed]}")
        return {"estimate_min": self.estimate_duration(plan).total_seconds() / 60.0}

    def execute_step(self, step: Step, cancel: threading.Event) -> dict:
        raise RuntimeError("RealRig not connected")

    def estimate_duration(self, plan: SynthesisPlan) -> timedelta:
        return timedelta(minutes=5)

