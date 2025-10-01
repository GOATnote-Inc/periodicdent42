from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import importlib
import json


class EventLogger:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.events_path = base_dir / "events.jsonl"
        self.measurements_path = base_dir / "measurements.parquet"
        self.measurements_jsonl = base_dir / "measurements.jsonl"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def log_measurement(self, campaign_id: str, step: int, results: Dict[str, Any]) -> None:
        record = {"campaign_id": campaign_id, "step": step, **results}
        with self.measurements_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        self._maybe_write_parquet()

    def _maybe_write_parquet(self) -> None:
        spec = importlib.util.find_spec("pyarrow")
        if spec is None:
            return
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        if not self.measurements_jsonl.exists():
            return
        rows = [json.loads(line) for line in self.measurements_jsonl.read_text().splitlines() if line.strip()]
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, self.measurements_path)
