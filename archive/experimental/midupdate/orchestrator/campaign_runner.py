from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from orchestrator.agent.planner import GlassBoxPlanner
from orchestrator.agent.constraint_checker import ConstraintChecker
from orchestrator.scheduler.surrogate import GPSurrogate
from orchestrator.storage.logger import EventLogger
from sims.transport_sim import TransportSimulator
from sims.xrd_sim import XRDSharpnessSimulator
from sims.synth_sim import SynthesisYieldSimulator


SIMULATORS = {
    "transport": TransportSimulator,
    "xrd": XRDSharpnessSimulator,
    "powder": SynthesisYieldSimulator,
}


@dataclass
class CampaignState:
    config: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_value: float = float("-inf")
    best_step: int = -1
    regret_curve: List[float] = field(default_factory=list)
    eig_curve: List[float] = field(default_factory=list)


class CampaignRunner:
    def __init__(
        self,
        planner: GlassBoxPlanner,
        constraint_checker: ConstraintChecker,
        surrogate: GPSurrogate,
        logger: EventLogger,
    ) -> None:
        self.planner = planner
        self.constraint_checker = constraint_checker
        self.surrogate = surrogate
        self.logger = logger

    def load_config(self, path: Path) -> Dict[str, Any]:
        return yaml.safe_load(path.read_text())

    def _initial_context(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        recent = history[-3:]
        return {"recent_runs": recent, "history_length": len(history)}

    def run(self, config_path: Path) -> CampaignState:
        config = self.load_config(config_path)
        campaign_id = config["id"]
        instrument = config["instrument"]
        simulator_cls = SIMULATORS[instrument]
        simulator = simulator_cls()
        state = CampaignState(config=config)
        baseline = 0.0
        for step in range(config["budget_steps"]):
            context = self._initial_context(state.history)
            constraints = config.get("constraints", {})
            plan_response = self.planner.propose(
                context=context,
                objective=config["objective"],
                constraints=constraints,
                candidate_space=config["search_space"],
            )
            _, repaired_plan, violations = self.constraint_checker.check(plan_response)
            proposed = repaired_plan["proposed_next"]
            params = proposed.get("value", {})
            measurement, metadata = simulator.run(params)
            duration = metadata.get("duration_min", 30.0)
            eig = self.surrogate.eig_per_hour(candidate_std=0.5, expected_duration_min=duration)
            self.surrogate.update(list(params.values()) or [0.0], measurement, duration)
            state.history.append(
                {
                    "step": step,
                    "plan": repaired_plan,
                    "measurement": measurement,
                    "metadata": metadata,
                    "violations": [violation.__dict__ for violation in violations],
                    "EIG_per_hour": eig,
                }
            )
            state.best_value = max(state.best_value, measurement)
            if state.best_value == measurement:
                state.best_step = step
            baseline = baseline if baseline else measurement
            regret = max(0.0, state.best_value - measurement)
            state.regret_curve.append(regret)
            state.eig_curve.append(eig)
            self.logger.log_event(
                "campaign_step",
                {
                    "campaign_id": campaign_id,
                    "step": step,
                    "measurement": measurement,
                    "best_so_far": state.best_value,
                    "regret": regret,
                    "eig_per_hour": eig,
                },
            )
            self.logger.log_measurement(campaign_id, step, metadata)
            if step - state.best_step >= config["stopping"]["plateau_patience"]:
                break
            improvement = state.best_value - baseline
            if improvement >= config["stopping"]["target_improvement"]:
                break
        self.logger.log_event(
            "campaign_complete",
            {
                "campaign_id": campaign_id,
                "best_value": state.best_value,
                "steps": len(state.history),
            },
        )
        return state


def summarize_campaign(state: CampaignState) -> Dict[str, Any]:
    return {
        "campaign": state.config["id"],
        "best_value": state.best_value,
        "steps": len(state.history),
        "regret_curve": state.regret_curve,
        "eig_curve": state.eig_curve,
        "glass_box_snapshots": [entry["plan"] for entry in state.history[:3]],
    }
