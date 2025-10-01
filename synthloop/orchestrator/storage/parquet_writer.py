from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None
import json

ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", "./data/runs"))
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def write_step_telemetry(run_id: str, step_index: int, telemetry: List[Dict]) -> Path:
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"step_{step_index:02d}.parquet"
    if pd is not None:
        df = pd.DataFrame(telemetry)
        df.to_parquet(path, index=False)
    else:  # fallback to JSON
        path.write_text(json.dumps(telemetry))
    return path


def write_bundle(run_id: str, files: Dict[str, Path], bundle_dir: Path | None = None) -> Path:
    bundle_dir = bundle_dir or (ARTIFACT_ROOT / run_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"{run_id}_bundle.zip"
    import zipfile

    with zipfile.ZipFile(bundle_path, "w") as zf:
        for name, path in files.items():
            zf.write(path, arcname=name)
    return bundle_path

