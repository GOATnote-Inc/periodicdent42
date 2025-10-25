from __future__ import annotations

from pathlib import Path

from orchestrator.agent.planner import GlassBoxPlanner, PlannerConfig, validate_plan

BASE_DIR = Path(__file__).resolve().parents[1]


def test_planner_generates_valid_plan():
    planner = GlassBoxPlanner(
        PlannerConfig(
            model_registry=BASE_DIR / "training" / "model_registry",
            dataset_dir=BASE_DIR / "data" / "processed",
        )
    )
    plan = planner.propose(
        context={"recent_runs": []},
        objective="maximize critical temperature",
        constraints={"max_temp_C": 900},
        candidate_space={"anneal_temp_C": [400, 900]},
    )
    valid, issues = validate_plan(plan)
    assert valid, f"Plan failed schema validation: {issues}"
    assert plan["proposed_next"]["value"]["anneal_temp_C"] <= 900
