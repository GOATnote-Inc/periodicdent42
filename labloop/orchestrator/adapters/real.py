from __future__ import annotations

import logging
import threading
from datetime import timedelta
from typing import Optional

from ..models.schemas import ExperimentPlan, SafetyLimits, Task, TaskType
from .base import Instrument, Measurement, PreflightReport

LOGGER = logging.getLogger(__name__)


class RealDevice(Instrument):
    """Thin wrapper around a hypothetical SCPI instrument.

    For this vertical slice we only validate configuration and log commands.
    Actual IO would use PyVISA or a vendor SDK.
    """

    def __init__(self, resource: str, safety_limits: SafetyLimits) -> None:
        self.resource = resource
        self.safety_limits = safety_limits

    def preflight(self, plan: ExperimentPlan) -> PreflightReport:  # type: ignore[override]
        try:
            self.safety_limits.ensure_complete()
        except ValueError as exc:  # pragma: no cover - defensive
            return PreflightReport(False, str(exc))
        if not plan.operator_ack:
            return PreflightReport(False, "Operator acknowledgement required for real hardware")
        return PreflightReport(True, f"Resource {self.resource} ready")

    def execute(self, task: Task, cancel_token: threading.Event) -> Measurement:  # type: ignore[override]
        LOGGER.info("[REAL] Executing task %s on %s", task.id, self.resource)
        if cancel_token.is_set():
            LOGGER.warning("Cancel requested before execution")
            return Measurement(task.id, {"status": "cancelled"})
        if task.type == TaskType.XRD:
            payload = {"status": "completed", "echo": task.parameters}
        else:
            payload = {"status": "completed", "echo": task.parameters}
        return Measurement(task.id, payload)

    def estimate_duration(self, task: Task) -> timedelta:  # type: ignore[override]
        duration = float(task.expected_duration_s or 10.0)
        return timedelta(seconds=duration)


def build_real_device(resource: str, safety_limits: Optional[SafetyLimits]) -> RealDevice:
    if safety_limits is None:
        raise ValueError("Safety limits are required for real device operation")
    return RealDevice(resource=resource, safety_limits=safety_limits)
