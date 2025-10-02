"""CLI entrypoint to execute the Phase 2 UV-Vis autonomous campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.lab.campaign import AutonomousCampaignRunner, get_campaign_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UV-Vis autonomous campaign simulation")
    parser.add_argument("--experiments", type=int, default=50, help="Number of experiments to execute")
    parser.add_argument("--hours", type=float, default=24.0, help="Maximum wall-clock hours to simulate")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("campaign_report.json"),
        help="Where to write the campaign summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner: AutonomousCampaignRunner = get_campaign_runner()
    report = runner.run_campaign(min_experiments=args.experiments, max_hours=args.hours)
    payload = {
        "campaign_id": report.campaign_id,
        "instrument": report.instrument_id,
        "experiments_requested": report.experiments_requested,
        "experiments_completed": report.experiments_completed,
        "failures": report.failures,
        "storage_uris": report.storage_uris,
        "started_at": report.started_at.isoformat(),
        "completed_at": report.completed_at.isoformat(),
    }
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Campaign {report.campaign_id} completed with {report.experiments_completed} experiments")


if __name__ == "__main__":
    main()
