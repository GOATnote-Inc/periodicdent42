from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .models.schemas import RunRecord


class BundleBuilder:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def build(self, record: RunRecord, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as archive:
            self._add_plan(record, archive)
            self._add_events(record, archive)
            self._add_results(record, archive)
            self._add_readme(record, archive)
            self._add_logs(record, archive)
        return output

    def _add_plan(self, record: RunRecord, archive: zipfile.ZipFile) -> None:
        plan_path = Path(record.plan_path)
        if plan_path.exists():
            archive.write(plan_path, arcname=f"{record.run_id}/plan.json")

    def _add_events(self, record: RunRecord, archive: zipfile.ZipFile) -> None:
        events_path = Path(record.events_path)
        if events_path.exists():
            archive.write(events_path, arcname=f"{record.run_id}/events.jsonl")

    def _add_results(self, record: RunRecord, archive: zipfile.ZipFile) -> None:
        results_dir = Path(record.results_path) / record.run_id
        if not results_dir.exists():
            return
        for file in results_dir.glob("**/*"):
            if file.is_file():
                archive.write(file, arcname=f"{record.run_id}/results/{file.name}")

    def _add_logs(self, record: RunRecord, archive: zipfile.ZipFile) -> None:
        log_path = Path(record.logs_path)
        if log_path.exists():
            archive.write(log_path, arcname=f"{record.run_id}/run.log")

    def _add_readme(self, record: RunRecord, archive: zipfile.ZipFile) -> None:
        readme = {
            "run_id": record.run_id,
            "backend": record.backend,
            "resource": record.resource,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "tasks": [task.dict() for task in record.plan.tasks],
        }
        archive.writestr(f"{record.run_id}/README.json", json.dumps(readme, indent=2))
