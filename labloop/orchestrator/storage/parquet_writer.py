from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


class ParquetWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_records(self, run_id: str, task_id: str, records: Iterable[Mapping[str, object]]) -> Path:
        df = pd.DataFrame(list(records))
        path = self.root / run_id / f"{task_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path
