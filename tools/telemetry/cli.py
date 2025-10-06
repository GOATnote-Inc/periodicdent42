from __future__ import annotations

import argparse
from typing import List

from services.telemetry.store import TelemetryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telemetry utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tail_parser = subparsers.add_parser("tail", help="Tail recent telemetry runs")
    tail_parser.add_argument("--last", type=int, default=20, help="Number of runs to display")

    return parser


def tail_runs(store: TelemetryStore, count: int) -> None:
    runs = store.list_runs(limit=count)
    for run in runs:
        summary = run.summary or {}
        answer_preview = summary.get("answer", "")[:80]
        print(
            f"{run.created_at.isoformat()} | {run.status:10s} | {run.id} | "
            f"{answer_preview}"
        )


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = TelemetryStore.from_env()

    if args.command == "tail":
        tail_runs(store, args.last)
    else:  # pragma: no cover
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
