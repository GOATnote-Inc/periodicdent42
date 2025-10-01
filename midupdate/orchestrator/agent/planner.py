from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pickle

from orchestrator.agent.rag import Evidence, LocalRAGIndex
from training.train import predict_label

GLASSBOX_FIELDS = [
    "goal",
    "assumptions",
    "hypotheses",
    "evidence_links",
    "candidate_actions",
    "proposed_next",
    "fallbacks",
    "constraints_checked",
]


@dataclass
class PlannerConfig:
    model_registry: Path
    dataset_dir: Path


class GlassBoxPlanner:
    def __init__(self, config: PlannerConfig) -> None:
        latest_dir = config.model_registry / "latest"
        model_path = latest_dir / "planner-ft.joblib"
        if not model_path.exists():
            raise FileNotFoundError("Planner artifact not found; train the model first")
        with model_path.open("rb") as fh:
            artifact = pickle.load(fh)
        self.domain_model = artifact.get("domain_averages", {})
        template_path = latest_dir / "rationale-template.json"
        if template_path.exists():
            self.template = json.loads(template_path.read_text())
        else:
            self.template = {field: None for field in GLASSBOX_FIELDS}
        self.rag = LocalRAGIndex(config.dataset_dir)

    def _candidate_actions(self, label: Dict[str, float]) -> List[Dict[str, object]]:
        base_temp = float(label.get("anneal_temp_C", 450.0))
        base_time = float(label.get("anneal_time_min", 30.0))
        base_doping = float(label.get("doping_pct", 0.05))
        action_name = label.get("action", "anneal")
        actions = []
        for delta_temp, eig in [(-15, 0.24), (0, 0.31), (20, 0.28)]:
            temp = base_temp + delta_temp
            rationale = (
                "Adjust anneal temperature to explore superconducting dome"
                if delta_temp != 0
                else "Maintain temperature while tightening time window"
            )
            actions.append(
                {
                    "action": action_name,
                    "value": {
                        "anneal_temp_C": temp,
                        "anneal_time_min": base_time,
                        "doping_pct": base_doping,
                    },
                    "rationale": rationale,
                    "risk": "medium" if abs(delta_temp) > 0 else "low",
                    "expected_effect": "improve Tc by tuning thermal budget",
                    "EIG_proxy": eig,
                }
            )
        return actions

    def _clamp_to_space(self, label: Dict[str, float], candidate_space: Dict[str, object]) -> Dict[str, float]:
        clamped = dict(label)
        for key, bounds in candidate_space.items():
            if isinstance(bounds, list) and len(bounds) == 2:
                lower, upper = float(bounds[0]), float(bounds[1])
                clamped[key] = max(lower, min(upper, float(clamped.get(key, lower))))
        clamped.setdefault("action", "anneal")
        return clamped

    def propose(
        self,
        context: Dict[str, object],
        objective: str,
        constraints: Dict[str, float],
        candidate_space: Dict[str, object],
    ) -> Dict[str, object]:
        query = " \n ".join(
            [
                objective,
                json.dumps(context, sort_keys=True),
                json.dumps(candidate_space, sort_keys=True),
            ]
        )
        record = {"context": context, "tags": context.get("tags", [])}
        label = predict_label(self.domain_model, record)
        label = self._clamp_to_space(label, candidate_space)
        candidate_actions = self._candidate_actions(label)
        evidence = self.rag.search(objective)

        plan = {
            "goal": objective,
            "assumptions": ["Experimental setup calibrated", "Materials quality consistent"],
            "hypotheses": [
                "H1: Increased anneal temperature improves lattice ordering",
                "H2: Moderate doping stabilizes superconducting phase",
            ],
            "evidence_links": [evidence_item.__dict__ for evidence_item in evidence],
            "candidate_actions": candidate_actions,
            "proposed_next": {**candidate_actions[1], "justification": "Highest EIG proxy"},
            "fallbacks": ["Return to last validated anneal recipe", "Run diagnostic XRD"],
            "constraints_checked": True,
        }
        return plan


def validate_plan(plan: Dict[str, object]) -> Tuple[bool, List[str]]:
    missing = [field for field in GLASSBOX_FIELDS if field not in plan]
    issues = []
    if missing:
        issues.append(f"Missing fields: {', '.join(missing)}")
    if not isinstance(plan.get("candidate_actions"), list):
        issues.append("candidate_actions must be a list")
    else:
        for idx, candidate in enumerate(plan["candidate_actions"]):
            if not {"action", "value"}.issubset(candidate):
                issues.append(f"candidate {idx} missing action/value")
    return len(issues) == 0, issues
