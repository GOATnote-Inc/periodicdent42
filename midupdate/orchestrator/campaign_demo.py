from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from orchestrator.agent.planner import GlassBoxPlanner, PlannerConfig
from orchestrator.agent.constraint_checker import ConstraintChecker
from orchestrator.campaign_runner import CampaignRunner, summarize_campaign
from orchestrator.scheduler.surrogate import GPSurrogate
from orchestrator.storage.logger import EventLogger

BASE_DIR = Path(__file__).resolve().parent.parent


def run_demo(config_dir: Path) -> List[Dict[str, object]]:
    planner = GlassBoxPlanner(PlannerConfig(model_registry=BASE_DIR / "training" / "model_registry", dataset_dir=BASE_DIR / "data" / "processed"))
    constraint_checker = ConstraintChecker(env_path=str(BASE_DIR / ".env"))
    surrogate = GPSurrogate()
    logger = EventLogger(BASE_DIR / "orchestrator" / "storage" / "artifacts")
    runner = CampaignRunner(planner, constraint_checker, surrogate, logger)

    summaries: List[Dict[str, object]] = []
    for name in ["tc_boost_v1.yaml", "xrd_sharpness_v1.yaml", "synth_yield_v1.yaml"]:
        config_path = config_dir / name
        state = runner.run(config_path)
        summaries.append(summarize_campaign(state))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulated campaigns")
    parser.add_argument("--config", type=str, default="campaigns", help="Directory with campaign YAML files")
    parser.add_argument("--output", type=str, default="campaigns/results.json")
    args = parser.parse_args()

    config_dir = Path(args.config)
    summaries = run_demo(config_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
