from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

EVENT_DIR = Path(os.environ.get("EVENT_LOG_DIR", "./data/events"))
EVENT_DIR.mkdir(parents=True, exist_ok=True)


def append_event(run_id: str, event_type: str, payload: Dict[str, Any]) -> Path:
    day_dir = EVENT_DIR / datetime.utcnow().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    log_path = day_dir / f"{run_id}.jsonl"
    record = {
        "ts": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "type": event_type,
        "payload": payload,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return log_path

