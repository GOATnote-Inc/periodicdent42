from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JsonlEventLog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, run_id: str, event: Dict[str, Any]) -> Path:
        day_folder = self.root / datetime.now().strftime("%Y-%m-%d")
        day_folder.mkdir(parents=True, exist_ok=True)
        path = day_folder / f"{run_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        return path
