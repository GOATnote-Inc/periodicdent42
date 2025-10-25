from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import yaml

from .base import BaseAdapter
from ..models.schemas import WorkflowEvent


class MockAdapter(BaseAdapter):
    """Generate deterministic workflow events from a seed and workflow config."""

    def __init__(
        self,
        workflow_config: Path,
        seed: int = 42,
        cycle_multiplier: float = 1.0,
        start: datetime | None = None,
    ):
        self.workflow_config = workflow_config
        self.seed = seed
        self.cycle_multiplier = cycle_multiplier
        self.start = start or (datetime.utcnow() - timedelta(days=14))
        self.steps = self._load_steps()

    def _load_steps(self) -> List[str]:
        config = yaml.safe_load(self.workflow_config.read_text())
        return [s["name"] for s in config.get("steps", [])]

    def stream(self) -> Iterable[WorkflowEvent]:
        rng = random.Random(self.seed)
        start = self.start
        unit_count = 120
        for idx in range(unit_count):
            unit_id = f"unit-{idx:04d}"
            cycle_scale = rng.uniform(0.8, 1.2) * self.cycle_multiplier
            enter_ts = start + timedelta(minutes=idx * 30)
            for step in self.steps:
                queue_delay = rng.uniform(5, 35) * cycle_scale
                work_time = rng.uniform(15, 60) * cycle_scale
            yield WorkflowEvent(
                ts=enter_ts,
                unit_id=unit_id,
                step=step,
                state="entered",
                attrs={"assignee": f"tech-{rng.randint(1,5)}"},
            )
            exit_ts = enter_ts + timedelta(minutes=queue_delay + work_time)
            yield WorkflowEvent(
                ts=exit_ts,
                unit_id=unit_id,
                step=step,
                state="exited",
                attrs={"assignee": f"tech-{rng.randint(1,5)}"},
            )
            enter_ts = exit_ts
        defect = None
        if rng.random() < 0.1:
            defect = rng.choice(["rework", "contamination", "instrument"])
        yield WorkflowEvent(
            ts=enter_ts,
            unit_id=unit_id,
            step="done",
            state="entered",
            attrs={"defect_code": defect},
        )
        yield WorkflowEvent(
            ts=enter_ts + timedelta(minutes=rng.uniform(1, 5)),
            unit_id=unit_id,
            step="done",
            state="exited",
            attrs={"defect_code": defect},
        )


def load_mock_adapter(
    workflow_config_path: Path,
    seed: int = 42,
    cycle_multiplier: float = 1.0,
    start: datetime | None = None,
) -> MockAdapter:
    return MockAdapter(workflow_config_path, seed, cycle_multiplier, start)
