#!/usr/bin/env python3
"""Utility for generating lightweight monitoring reports for CI workflows."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def iso_timestamp() -> str:
    """Return a UTC timestamp in ISO 8601 format without microseconds."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def safe_git_command(args: List[str]) -> Optional[str]:
    """Run a git command and return its stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def line_count(files: Iterable[Path]) -> int:
    """Count total lines across the provided files."""
    total = 0
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                total += sum(1 for _ in handle)
        except OSError:
            continue
    return total


def detect_health_endpoints(api_path: Path) -> Dict[str, bool]:
    """Inspect the API module for health endpoints."""
    details = {"health": False, "healthz": False}
    if not api_path.exists():
        return details

    try:
        contents = api_path.read_text(encoding="utf-8")
    except OSError:
        return details

    details["health"] = "/health" in contents
    details["healthz"] = "/healthz" in contents
    return details


def extract_function_names(module_path: Path) -> List[str]:
    """Extract top-level function names from a Python module."""
    if not module_path.exists():
        return []

    try:
        contents = module_path.read_text(encoding="utf-8")
    except OSError:
        return []

    pattern = re.compile(r"^def\s+([a-zA-Z_][\w]*)\(", re.MULTILINE)
    return sorted(set(pattern.findall(contents)))


def read_service_metadata() -> Dict[str, Optional[str]]:
    """Infer service metadata from config files and workflows."""
    metadata: Dict[str, Optional[str]] = {
        "project_id": None,
        "region": None,
        "service_name": None,
    }

    config_path = REPO_ROOT / "configs" / "service.yaml"
    if config_path.exists():
        try:
            for line in config_path.read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip() or None
                if key in ("project", "project_id"):
                    metadata["project_id"] = value
                elif key == "region":
                    metadata["region"] = value
                elif key in ("service", "service_name"):
                    metadata["service_name"] = value
        except OSError:
            pass

    workflow_path = REPO_ROOT / ".github" / "workflows" / "cicd.yaml"
    if workflow_path.exists():
        try:
            contents = workflow_path.read_text(encoding="utf-8")
        except OSError:
            contents = ""

        project_match = re.search(r"PROJECT_ID:\s*([A-Za-z0-9_-]+)", contents)
        region_match = re.search(r"REGION:\s*([A-Za-z0-9-]+)", contents)
        service_match = re.search(r"SERVICE_NAME:\s*([A-Za-z0-9_-]+)", contents)
        metadata.setdefault("project_id", None)
        metadata.setdefault("region", None)
        metadata.setdefault("service_name", None)
        if metadata["project_id"] is None and project_match:
            metadata["project_id"] = project_match.group(1)
        if metadata["region"] is None and region_match:
            metadata["region"] = region_match.group(1)
        if metadata["service_name"] is None and service_match:
            metadata["service_name"] = service_match.group(1)

    return metadata


def collect_repo_snapshot() -> Dict[str, object]:
    """Gather snapshot metrics about the repository."""
    src_dir = REPO_ROOT / "app" / "src"
    tests_dir = REPO_ROOT / "app" / "tests"
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    api_module = src_dir / "api" / "main.py"
    metrics_module = src_dir / "monitoring" / "metrics.py"

    python_files = [p for p in src_dir.rglob("*.py") if "__pycache__" not in p.parts]
    test_files = [p for p in tests_dir.rglob("test_*.py") if "__pycache__" not in p.parts]
    workflow_files = [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]

    return {
        "generated_at": iso_timestamp(),
        "python_module_count": len(python_files),
        "python_line_count": line_count(python_files),
        "test_file_count": len(test_files),
        "test_line_count": line_count(test_files),
        "workflow_count": len(workflow_files),
        "workflow_files": sorted(path.name for path in workflow_files),
        "health_endpoints": detect_health_endpoints(api_module),
        "monitoring_functions": extract_function_names(metrics_module),
        "service_metadata": read_service_metadata(),
        "latest_commit": safe_git_command(["rev-parse", "HEAD"]),
        "latest_commit_subject": safe_git_command(["log", "-1", "--pretty=%s"]),
    }


def build_performance_report(snapshot: Dict[str, object]) -> Dict[str, object]:
    return {
        "kind": "performance",
        "status": "ok",
        "generated_at": snapshot["generated_at"],
        "summary": "Repository performance snapshot derived from source metrics.",
        "metrics": {
            "python_modules": snapshot["python_module_count"],
            "python_loc": snapshot["python_line_count"],
            "test_modules": snapshot["test_file_count"],
            "test_loc": snapshot["test_line_count"],
            "workflows": snapshot["workflow_count"],
        },
        "git": {
            "commit": snapshot.get("latest_commit"),
            "subject": snapshot.get("latest_commit_subject"),
        },
    }


def build_uptime_report(snapshot: Dict[str, object]) -> Dict[str, object]:
    health_endpoints = snapshot["health_endpoints"]
    metadata = snapshot["service_metadata"]
    status = "ok" if any(health_endpoints.values()) else "attention"

    return {
        "kind": "uptime",
        "status": status,
        "generated_at": snapshot["generated_at"],
        "service": metadata,
        "checks": {
            "health_endpoint_present": health_endpoints["health"],
            "healthz_endpoint_present": health_endpoints["healthz"],
        },
    }


def build_health_report(snapshot: Dict[str, object]) -> Dict[str, object]:
    health_endpoints = snapshot["health_endpoints"]
    tests_available = snapshot["test_file_count"] > 0
    status = "ok" if tests_available and any(health_endpoints.values()) else "attention"

    return {
        "kind": "health",
        "status": status,
        "generated_at": snapshot["generated_at"],
        "summary": "Health coverage derived from endpoint availability and tests.",
        "coverage": {
            "tests": snapshot["test_file_count"],
            "health_endpoint": health_endpoints["health"],
            "healthz_endpoint": health_endpoints["healthz"],
        },
    }


def build_metrics_report(snapshot: Dict[str, object]) -> Dict[str, object]:
    monitoring_functions = snapshot["monitoring_functions"]
    status = "ok" if monitoring_functions else "attention"

    return {
        "kind": "metrics",
        "status": status,
        "generated_at": snapshot["generated_at"],
        "summary": "Available monitoring hooks discovered in the codebase.",
        "functions": monitoring_functions,
    }


def build_alert_report(reports: List[Dict[str, object]]) -> Dict[str, object]:
    statuses = [report.get("status", "unknown") for report in reports]
    degraded = [status for status in statuses if status not in ("ok", "pass")]
    overall_status = "ok" if not degraded else "attention"

    return {
        "kind": "alert",
        "status": overall_status,
        "generated_at": iso_timestamp(),
        "summary": "Aggregated status derived from monitoring sub-reports.",
        "sources": reports,
    }


def write_report(report: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monitoring report data")
    parser.add_argument(
        "--kind",
        choices=["performance", "uptime", "health", "metrics", "alert"],
        required=True,
        help="Type of report to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the report JSON to",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[],
        help="Input report paths when generating alert summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "alert":
        if not args.inputs:
            raise SystemExit("Alert reports require at least one --inputs value")
        reports: List[Dict[str, object]] = []
        for input_path in args.inputs:
            try:
                with input_path.open("r", encoding="utf-8") as handle:
                    reports.append(json.load(handle))
            except (OSError, json.JSONDecodeError):
                continue
        if not reports:
            raise SystemExit("No readable input reports were provided")
        report = build_alert_report(reports)
        write_report(report, args.output)
        return

    snapshot = collect_repo_snapshot()
    builders = {
        "performance": build_performance_report,
        "uptime": build_uptime_report,
        "health": build_health_report,
        "metrics": build_metrics_report,
    }
    report_builder = builders[args.kind]
    report = report_builder(snapshot)
    write_report(report, args.output)


if __name__ == "__main__":
    main()
