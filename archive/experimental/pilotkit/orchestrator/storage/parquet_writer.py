from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class ParquetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: Iterable[dict]) -> None:
        df = pd.DataFrame(list(records))
        if df.empty:
            return
        df.to_parquet(self.path, index=False)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.path)
