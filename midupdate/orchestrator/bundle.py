from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copy2

BASE_DIR = Path(__file__).resolve().parent.parent


def export_bundle(run_id: str, output_dir: Path) -> Path:
    bundle_dir = output_dir / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    events_path = BASE_DIR / "orchestrator" / "storage" / "artifacts" / "events.jsonl"
    measurements_path = BASE_DIR / "orchestrator" / "storage" / "artifacts" / "measurements.parquet"
    if events_path.exists():
        copy2(events_path, bundle_dir / "events.jsonl")
    if measurements_path.exists():
        copy2(measurements_path, bundle_dir / "measurements.parquet")

    latest_dir = BASE_DIR / "training" / "model_registry" / "latest"
    for name in ["planner-ft.joblib", "metrics.json", "model_card.md", "train_config.yaml"]:
        path = latest_dir / name
        if path.exists():
            copy2(path, bundle_dir / name)

    metadata = {
        "run_id": run_id,
        "model_dir": str(latest_dir),
        "events": str(bundle_dir / "events.jsonl"),
        "measurements": str(bundle_dir / "measurements.parquet"),
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export provenance bundle")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default="bundles")
    args = parser.parse_args()

    output_dir = Path(args.output)
    bundle_dir = export_bundle(args.run_id, output_dir)
    print(f"Bundle exported to {bundle_dir}")


if __name__ == "__main__":
    main()
