from __future__ import annotations

import math
import os
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List

from ..models.plan import Step, SynthesisPlan


class SimRig:
    def __init__(self):
        self.fault_mode = os.environ.get("SIM_FAULT_MODE", "none")
        self.interlocks = {
            "EnclosureClosed": True,
            "EStopNotEngaged": True,
            "VentilationOn": True,
            "ScaleHealthy": True,
            "PowerOK": True,
        }

    def preflight(self, plan: SynthesisPlan) -> Dict:
        return {"estimate_min": self.estimate_duration(plan).total_seconds() / 60.0}

    def execute_step(self, step: Step, cancel: threading.Event) -> Dict:
        telemetry: List[Dict] = []
        start = datetime.utcnow()
        duration = 1.0
        status = "success"
        detail = {}

        if step.type == "Dispense":
            actual = step.mass_g + random.uniform(-0.02, 0.02)
            telemetry.append(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "setpoint_g": step.mass_g,
                    "measured_g": actual,
                }
            )
            detail["measured_g"] = actual
        elif step.type == "Mix":
            duration = step.duration_s / 60.0
            rpm = step.rpm + random.uniform(-5, 5)
            for i in range(10):
                if cancel.is_set():
                    status = "aborted"
                    break
                telemetry.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "rpm": rpm + random.uniform(-2, 2),
                    }
                )
                time.sleep(duration / 10)
            detail["avg_rpm"] = rpm
        elif step.type == "Heat":
            duration = (step.hold_min or 1) * 0.6
            overshoot = 0
            if self.fault_mode == "heater_overshoot":
                overshoot = random.uniform(20, 40)
            for i in range(10):
                if cancel.is_set():
                    status = "aborted"
                    break
                temp = step.target_C + overshoot * math.exp(-0.2 * i) + random.uniform(-2, 2)
                telemetry.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "temp_C": temp,
                        "setpoint_C": step.target_C,
                    }
                )
                time.sleep(duration / 10)
            detail["max_temp"] = max(t["temp_C"] for t in telemetry) if telemetry else step.target_C
            if overshoot > 0:
                status = "failed"
                detail["error"] = "heater overshoot"
        elif step.type == "Cool":
            for i in range(5):
                if cancel.is_set():
                    status = "aborted"
                    break
                telemetry.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "temp_C": step.target_C + random.uniform(-1, 1),
                    }
                )
                time.sleep(0.1)
        elif step.type == "Collect":
            telemetry.append(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "sample_id": step.sample_id,
                }
            )

        if cancel.is_set() and status != "failed":
            status = "aborted"

        end = datetime.utcnow()
        return {
            "telemetry": telemetry,
            "status": status,
            "detail": detail,
            "started_at": start,
            "ended_at": end,
        }

    def estimate_duration(self, plan: SynthesisPlan) -> timedelta:
        total = 0
        for step in plan.steps:
            if step.type == "Mix":
                total += step.duration_s
            elif step.type == "Heat":
                total += (step.hold_min or 1) * 60
            else:
                total += 30
        return timedelta(seconds=total)

