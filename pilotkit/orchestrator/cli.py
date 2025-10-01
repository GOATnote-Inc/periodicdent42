from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import typer
import yaml

from .adapters.mock_adapter import load_mock_adapter
from .analytics.metrics import MetricEmitter, split_baseline_pilot
from .analytics.value_stream import summarize_metrics
from .feedback.processor import FeedbackProcessor
from .iteration.planner import export_plan, rank_backlog
from .models.schemas import FeedbackItem, IterationPlanRequest
from .playbook.generator import generate_playbook
from .report.generator import PilotReportGenerator
from .storage.db import Database, FeedbackRow, MetricRow
from .storage.eventlog import EventLog
from .storage.parquet_writer import ParquetWriter

app = typer.Typer()

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
EVENT_LOG = EventLog(DATA_DIR / "events.jsonl")
PARQUET_WRITER = ParquetWriter(DATA_DIR / "metrics.parquet")
DATABASE = Database(DATA_DIR / "pilot.db")
WORKFLOW_CONFIG = ROOT_DIR / "configs" / "workflow.sample.yaml"
PLAYBOOK_DIR = ROOT_DIR / "reports"
PLAYBOOK_DIR.mkdir(exist_ok=True, parents=True)
PROCESSOR = FeedbackProcessor()


def load_steps() -> List[str]:
    data = yaml.safe_load(WORKFLOW_CONFIG.read_text())
    return [step["name"] for step in data.get("steps", [])]


def reset_state() -> None:
    global DATABASE
    for file in [EVENT_LOG.path, PARQUET_WRITER.path, DATA_DIR / "pilot.db"]:
        if file.exists():
            file.unlink()
    DATABASE = Database(DATA_DIR / "pilot.db")


@app.command()
def demo(seed: int = 42, cycle_reduction: float = 0.2):
    """Load sample configs, simulate baseline + pilot events, and compute metrics."""

    reset_state()
    steps = load_steps()
    baseline_start = datetime.utcnow() - timedelta(days=14)
    pilot_start = baseline_start + timedelta(days=7)
    baseline_adapter = load_mock_adapter(
        WORKFLOW_CONFIG, seed=seed, cycle_multiplier=1.0, start=baseline_start
    )
    pilot_adapter = load_mock_adapter(
        WORKFLOW_CONFIG,
        seed=seed + 1,
        cycle_multiplier=max(0.1, 1.0 - cycle_reduction),
        start=pilot_start,
    )
    events = list(baseline_adapter.stream()) + list(pilot_adapter.stream())
    EVENT_LOG.append(events)
    emitter = MetricEmitter(steps)
    records = emitter.emit(events)
    PARQUET_WRITER.write(records)
    metric_rows = [
        MetricRow(
            unit_id=r["unit_id"],
            period=datetime.fromisoformat(r["period"]),
            cycle_time_s=r["cycle_time_s"],
            touch_time_s=r["touch_time_s"],
            wait_time_s=r["wait_time_s"],
            yield_ok=r["yield_ok"],
            defect_code=r["defect_code"],
        )
        for r in records
    ]
    DATABASE.replace_metrics(metric_rows)
    baseline, pilot = split_baseline_pilot(records, pilot_start)
    baseline_summary = summarize_metrics(baseline)
    pilot_summary = summarize_metrics(pilot)
    typer.echo(f"Baseline cycle p50: {baseline_summary['cycle_time_p50']:.1f}s")
    typer.echo(f"Pilot cycle p50: {pilot_summary['cycle_time_p50']:.1f}s")
    typer.echo(f"Yield delta: {(pilot_summary['yield_rate'] - baseline_summary['yield_rate']) * 100:.1f}%")
    generate_playbook(ROOT_DIR / "configs" / "pilot.sample.yaml", PLAYBOOK_DIR)
    typer.echo("Demo complete. Data written to data/ directory.")


@app.command()
def report():
    records = PARQUET_WRITER.read().to_dict("records")
    if not records:
        typer.echo("No metrics found. Run demo first.")
        raise typer.Exit(code=1)
    pilot_start = datetime.utcnow() - timedelta(days=7)
    baseline, pilot = split_baseline_pilot(records, pilot_start)
    generator = PilotReportGenerator(PLAYBOOK_DIR)
    result = generator.generate(baseline, pilot, "Pilot Impact Report")
    typer.echo(f"Report markdown: {result['markdown']}")
    typer.echo(f"Chart: {result['chart']}")


@app.command()
def iteration():
    records = PARQUET_WRITER.read().to_dict("records")
    if not records:
        typer.echo("No metrics found. Run demo first.")
        raise typer.Exit(code=1)
    df = PARQUET_WRITER.read()
    if df.empty:
        raise typer.Exit(code=1)
    pilot_start = datetime.utcnow() - timedelta(days=7)
    baseline, pilot = split_baseline_pilot(records, pilot_start)
    if not baseline or not pilot:
        raise typer.Exit(code=1)
    baseline_median = pd.Series([r["cycle_time_s"] for r in baseline]).median()
    pilot_median = pd.Series([r["cycle_time_s"] for r in pilot]).median()
    cycle_pct = ((pilot_median - baseline_median) / max(baseline_median, 1)) * 100
    request = IterationPlanRequest(
        metrics_delta={"cycle_time_pct": float(cycle_pct)},
        top_feedback_themes=["Cycle bottleneck", "Quality escapes", "Usability training"],
        guardrails={"yield_rate": "Maintain > 0.95"},
    )
    plan = rank_backlog(request)
    files = export_plan(plan, PLAYBOOK_DIR)
    typer.echo(f"Iteration plan written to {files['markdown']}")


@app.command()
def seed_feedback(count: int = 10):
    now = datetime.utcnow()
    items = []
    severities = ["P0", "P1", "P2", "P3"]
    steps = ["queue", "work", "review"]
    for idx in range(count):
        items.append(
            FeedbackItem(
                ts=now - timedelta(minutes=idx * 5),
                severity=severities[idx % len(severities)],
                step=steps[idx % len(steps)],
                text=f"Issue {idx} slow response on step {steps[idx % len(steps)]}",
                frustration=3 + (idx % 2),
                task_success=idx % 3 != 0,
                time_on_task_s=600 + idx * 10,
            )
        )
    themes = PROCESSOR.aggregate_insights(items)
    rows = []
    for item in items:
        tags = PROCESSOR.auto_tag(item)
        rows.append(
            FeedbackRow(
                ts=item.ts,
                step=item.step,
                severity=item.severity,
                tags=",".join(tags),
                text=PROCESSOR._redact(item.text),
                frustration=item.frustration,
                task_success=item.task_success,
                time_on_task_s=item.time_on_task_s,
                theme=tags[0] if tags else None,
            )
        )
    DATABASE.add_feedback(rows)
    typer.echo(f"Seeded {len(items)} feedback items into the database.")
    typer.echo(f"Themes: {themes}")


if __name__ == "__main__":
    app()
