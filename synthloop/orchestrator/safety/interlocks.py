from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

from ..models.plan import SynthesisPlan


@dataclass
class InterlockStatus:
    name: str
    ok: bool
    detail: str


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def check_plan_limits(plan: SynthesisPlan) -> List[InterlockStatus]:
    statuses: List[InterlockStatus] = []
    max_t = env_float("MAX_T_C", 750)
    max_ramp = env_float("MAX_RAMP_C_PER_MIN", 10)
    max_rpm = env_float("MAX_RPM", 600)
    max_batch = env_float("MAX_BATCH_G", 20)
    total_mass = sum(r.get("target_g", 0) for r in plan.reagents)
    statuses.append(InterlockStatus("BatchMass", total_mass <= max_batch, f"total {total_mass}g <= {max_batch}"))
    for idx, step in enumerate(plan.steps):
        if step.type == "Heat":
            statuses.append(
                InterlockStatus(
                    f"HeatTemp_{idx}",
                    step.target_C <= max_t,
                    f"target {step.target_C} <= {max_t}",
                )
            )
            if step.ramp_C_per_min is not None:
                statuses.append(
                    InterlockStatus(
                        f"HeatRamp_{idx}",
                        step.ramp_C_per_min <= max_ramp,
                        f"ramp {step.ramp_C_per_min} <= {max_ramp}",
                    )
                )
        if step.type == "Mix" and step.rpm is not None:
            statuses.append(InterlockStatus(f"MixRPM_{idx}", step.rpm <= max_rpm, f"rpm {step.rpm} <= {max_rpm}"))
    return statuses


def check_operational_interlocks(state: Dict[str, bool]) -> List[InterlockStatus]:
    required = {
        "EnclosureClosed": state.get("EnclosureClosed", True),
        "EStopNotEngaged": state.get("EStopNotEngaged", True),
        "VentilationOn": (not bool(os.environ.get("VENT_REQUIRED", "true").lower() == "true"))
        or state.get("VentilationOn", True),
        "ScaleHealthy": state.get("ScaleHealthy", True),
        "PowerOK": state.get("PowerOK", True),
    }
    return [InterlockStatus(name, ok, "bool") for name, ok in required.items()]


def calibrations_fresh(calibration_refs: Dict[str, str]) -> List[InterlockStatus]:
    statuses: List[InterlockStatus] = []
    now = datetime.utcnow()
    scale_last = datetime.fromisoformat(calibration_refs["last_calibrated"])
    temp_last = datetime.fromisoformat(calibration_refs["last_calibrated_temp"])
    window = timedelta(days=30)
    statuses.append(InterlockStatus("ScaleCalibrationFresh", now - scale_last <= window, f"{scale_last.isoformat()}"))
    statuses.append(InterlockStatus("TempProbeCalibrationFresh", now - temp_last <= window, f"{temp_last.isoformat()}"))
    return statuses


def verify(plan: SynthesisPlan, interlock_state: Dict[str, bool]) -> List[InterlockStatus]:
    statuses = []
    statuses.extend(check_plan_limits(plan))
    statuses.extend(check_operational_interlocks(interlock_state))
    statuses.extend(calibrations_fresh(plan.calibration_refs))
    return statuses


