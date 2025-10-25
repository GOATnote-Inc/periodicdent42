from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None

from ..models.plan import SynthesisPlan


class QCRuleResult(Dict[str, Any]):
    pass


def qc_plan_completeness(plan: SynthesisPlan) -> QCRuleResult:
    status = "pass"
    evidence = {"steps": len(plan.steps), "reagents": len(plan.reagents)}
    if len(plan.steps) == 0:
        status = "fail"
    return {"name": "PlanCompleteness", "status": status, "evidence": evidence}


def qc_mass_balance(plan: SynthesisPlan, step_summaries: List[Dict[str, Any]]) -> QCRuleResult:
    target = sum(r.get("target_g", 0) for r in plan.reagents)
    measured = sum(s.get("detail", {}).get("measured_g", 0) for s in step_summaries if s["step"].type == "Dispense")
    tolerance = plan.sample.get("tolerance_g", 0.1)
    diff = abs(target - measured)
    status = "pass" if diff <= tolerance and measured <= plan.sample.get("batch_g", target) + tolerance else "fail"
    evidence = {"target": target, "measured": measured, "diff": diff}
    return {"name": "MassBalance", "status": status, "evidence": evidence}


def qc_temperature(step_summaries: List[Dict[str, Any]]) -> QCRuleResult:
    status = "pass"
    evidence = {}
    for summary in step_summaries:
        if summary["step"].type == "Heat":
            max_temp = summary["detail"].get("max_temp", summary["step"].target_C)
            evidence[summary["step"].target_C] = max_temp
            if max_temp - summary["step"].target_C > 15:
                status = "fail"
    return {"name": "ProcessAdherence", "status": status, "evidence": evidence}


def qc_telemetry(run_artifact_dir: Path) -> QCRuleResult:
    status = "pass"
    evidence = {}
    for parquet in run_artifact_dir.glob("step_*.parquet"):
        if pd is not None:
            df = pd.read_parquet(parquet)
            if df.empty:
                status = "warn"
                evidence[str(parquet)] = "empty"
            else:
                if "ts" in df.columns and not df["ts"].is_monotonic_increasing:
                    status = "fail"
                    evidence[str(parquet)] = "non-monotonic timestamps"
        else:
            try:
                data = json.loads(parquet.read_text())
                if not isinstance(data, list) or not data:
                    status = "warn"
                    evidence[str(parquet)] = "missing telemetry"
            except Exception:  # noqa: BLE001
                status = "fail"
                evidence[str(parquet)] = "unreadable"
    return {"name": "TelemetryIntegrity", "status": status, "evidence": evidence}


def qc_environment(interlock_events: List[Dict[str, Any]]) -> QCRuleResult:
    status = "pass"
    violations = [evt for evt in interlock_events if not evt.get("ok", True)]
    if violations:
        status = "fail"
    return {"name": "EnvironmentInterlocks", "status": status, "evidence": violations}


def run_qc(plan: SynthesisPlan, step_summaries: List[Dict[str, Any]], run_artifact_dir: Path, interlock_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    rules = [
        qc_plan_completeness(plan),
        qc_mass_balance(plan, step_summaries),
        qc_temperature(step_summaries),
        qc_telemetry(run_artifact_dir),
        qc_environment(interlock_events),
    ]
    statuses = [rule["status"] for rule in rules]
    overall = "pass"
    if "fail" in statuses:
        overall = "fail"
    elif "warn" in statuses:
        overall = "warn"
    return {"overall": overall, "rules": rules, "created_at": datetime.utcnow()}

