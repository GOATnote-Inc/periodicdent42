from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dotenv import dotenv_values


@dataclass
class ConstraintViolation:
    parameter: str
    value: float
    limit: float


class ConstraintChecker:
    def __init__(self, env_path: str | None = None) -> None:
        env_values = dotenv_values(env_path or ".env")
        self.limits = {
            "anneal_temp_C": float(env_values.get("MAX_TEMP_C", 900)),
            "anneal_time_min": float(env_values.get("MAX_TIME_MIN", 180)),
            "ramp_rate": float(env_values.get("MAX_RAMP_C_PER_MIN", 10)),
            "voltage_V": float(env_values.get("MAX_VOLTAGE_V", 5)),
            "batch_size": float(env_values.get("MAX_BATCH_SIZE", 8)),
        }

    def _clamp(self, parameter: str, value: float) -> float:
        if parameter == "anneal_temp_C":
            return min(value, self.limits["anneal_temp_C"])
        if parameter == "anneal_time_min":
            return min(value, self.limits["anneal_time_min"])
        if parameter == "doping_pct":
            return max(0.0, min(value, 0.25))
        return value

    def check(self, plan: Dict[str, object]) -> Tuple[bool, Dict[str, object], List[ConstraintViolation]]:
        violations: List[ConstraintViolation] = []
        repaired_plan = plan.copy()
        repaired_candidates = []
        for candidate in plan.get("candidate_actions", []):
            candidate_copy = {**candidate}
            values = dict(candidate.get("value", {}))
            for key, value in values.items():
                if key == "anneal_temp_C" and value > self.limits["anneal_temp_C"]:
                    violations.append(
                        ConstraintViolation(parameter=key, value=float(value), limit=self.limits["anneal_temp_C"])
                    )
                    values[key] = self._clamp(key, float(value))
                if key == "anneal_time_min" and value > self.limits["anneal_time_min"]:
                    violations.append(
                        ConstraintViolation(parameter=key, value=float(value), limit=self.limits["anneal_time_min"])
                    )
                    values[key] = self._clamp(key, float(value))
                if key == "doping_pct":
                    clamped = self._clamp(key, float(value))
                    if clamped != value:
                        violations.append(ConstraintViolation(parameter=key, value=float(value), limit=0.25))
                        values[key] = clamped
            candidate_copy["value"] = values
            repaired_candidates.append(candidate_copy)
        repaired_plan["candidate_actions"] = repaired_candidates
        proposed = dict(plan.get("proposed_next", {}))
        if proposed:
            proposed_values = dict(proposed.get("value", {}))
            for key, value in proposed_values.items():
                proposed_values[key] = self._clamp(key, float(value))
            proposed["value"] = proposed_values
            repaired_plan["proposed_next"] = proposed
        return len(violations) == 0, repaired_plan, violations
