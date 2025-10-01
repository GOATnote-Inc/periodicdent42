from __future__ import annotations

from pathlib import Path

from orchestrator.agent.constraint_checker import ConstraintChecker
from orchestrator.agent.planner import GlassBoxPlanner, PlannerConfig
from orchestrator.campaign_runner import CampaignRunner
from orchestrator.scheduler.surrogate import GPSurrogate
from orchestrator.storage.logger import EventLogger

BASE_DIR = Path(__file__).resolve().parents[1]


def test_campaign_simulation_runs(tmp_path):
    planner = GlassBoxPlanner(
        PlannerConfig(
            model_registry=BASE_DIR / "training" / "model_registry",
            dataset_dir=BASE_DIR / "data" / "processed",
        )
    )
    checker = ConstraintChecker(env_path=str(BASE_DIR / ".env"))
    surrogate = GPSurrogate()
    logger = EventLogger(tmp_path)
    runner = CampaignRunner(planner, checker, surrogate, logger)
    config_path = BASE_DIR / "campaigns" / "tc_boost_v1.yaml"
    state = runner.run(config_path)
    assert state.best_value > 0
    assert state.history, "Campaign history should not be empty"
