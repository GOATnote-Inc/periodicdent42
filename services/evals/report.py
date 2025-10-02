from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from services.evals.runner import EvalRun


@dataclass
class EvalReport:
    run_id: str
    summary_path: Path
    json_path: Path


def save_report(eval_run: EvalRun, output_dir: Path) -> EvalReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{eval_run.run_id}.json"
    md_path = output_dir / f"{eval_run.run_id}.md"
    json_path.write_text("{}")
    md_path.write_text("# Eval Summary\n\nSynthetic placeholder.")
    return EvalReport(run_id=eval_run.run_id, summary_path=md_path, json_path=json_path)


__all__ = ["save_report", "EvalReport"]
