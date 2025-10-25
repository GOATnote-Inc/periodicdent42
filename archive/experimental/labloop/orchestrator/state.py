from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

from .adapters.base import Instrument
from .models.schemas import (
    ExperimentPlan,
    MeasurementResult,
    RunRecord,
    RunStatus,
    SchedulerDecision,
    Task,
)
from .scheduler.eig_scheduler import EIGScheduler
from .storage.jsonl_eventlog import JsonlEventLog
from .storage.parquet_writer import ParquetWriter

LOGGER = logging.getLogger(__name__)


class RunState:
    def __init__(
        self,
        record: RunRecord,
        instrument: Instrument,
        scheduler: EIGScheduler,
        event_log: JsonlEventLog,
        parquet_writer: ParquetWriter,
    ) -> None:
        self.record = record
        self.instrument = instrument
        self.scheduler = scheduler
        self.event_log = event_log
        self.parquet_writer = parquet_writer
        self.pending: Deque[Task] = deque(record.plan.tasks)
        self.completed: List[MeasurementResult] = []
        self.logs: List[str] = []
        self.cancel_token = threading.Event()
        self.last_decision: Optional[SchedulerDecision] = None

    def log(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        LOGGER.info("%s %s", self.record.run_id, message)
        if self.record.logs_path:
            path = Path(self.record.logs_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(entry + "\n")


class OrchestratorState:
    def __init__(self, data_dir: Path) -> None:
        self.runs: Dict[str, RunState] = {}
        self._lock = threading.Lock()
        self.data_dir = data_dir
        self.events = JsonlEventLog(data_dir / "data" / "events")
        self.parquet = ParquetWriter(data_dir / "data" / "results")
        self.scheduler = EIGScheduler()

    def create_run(
        self,
        run_record: RunRecord,
        instrument: Instrument,
    ) -> RunState:
        with self._lock:
            state = RunState(
                record=run_record,
                instrument=instrument,
                scheduler=self.scheduler,
                event_log=self.events,
                parquet_writer=self.parquet,
            )
            self.runs[run_record.run_id] = state
            return state

    def get(self, run_id: str) -> Optional[RunState]:
        with self._lock:
            return self.runs.get(run_id)

    def list_runs(self) -> List[RunRecord]:
        with self._lock:
            return [state.record for state in self.runs.values()]
