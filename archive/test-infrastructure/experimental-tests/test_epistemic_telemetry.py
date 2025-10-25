"""Unit tests for telemetry plumbing (CI ingestion + ledger emission)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.collect_ci_runs import load_pytest_results
from scripts.gen_ci_report import (
    compute_metrics,
    emit_experiment_ledger,
)


def write_pytest_report(tmp_path: Path) -> Path:
    report = {
        "tests": [
            {
                "nodeid": "tests/test_materials.py::test_lattice_stability",
                "outcome": "passed",
                "when": "call",
                "duration": 2.5,
                "start": "2024-01-01T00:00:00Z",
                "metadata": {
                    "domain": "materials",
                    "suite": "materials",
                    "model_uncertainty": 0.3,
                    "entropy_before": 0.85,
                    "entropy_after": 0.40,
                    "metrics": {"convergence_error": 1e-5},
                },
            },
            {
                "nodeid": "tests/test_robotics.py::test_path_planning",
                "outcome": "failed",
                "when": "call",
                "duration": 4.0,
                "start": "2024-01-01T00:05:00Z",
                "metadata": {
                    "domain": "robotics",
                    "suite": "robotics",
                    "model_uncertainty": 0.7,
                    "failure_type": "timeout",
                },
            },
        ]
    }

    path = tmp_path / "pytest-report.json"
    path.write_text(json.dumps(report))
    return path


def test_load_pytest_results_extracts_entropy(tmp_path: Path) -> None:
    report_path = write_pytest_report(tmp_path)
    tests, total_duration, total_cost = load_pytest_results(report_path, runner_usd_per_hour=0.6)

    assert len(tests) == 2
    first = tests[0]
    assert first["name"].endswith("test_lattice_stability")
    assert pytest.approx(first["duration_sec"], rel=1e-6) == 2.5
    assert first["entropy_before"] == 0.85
    assert first["entropy_after"] == 0.40
    assert tests[1]["failure_type"] == "timeout"
    # 2.5 + 4.0 seconds total duration, ensure cost conversion from seconds to USD/h
    assert pytest.approx(total_duration, rel=1e-6) == 6.5
    assert total_cost > 0


def test_compute_metrics_exposes_budget_utilization() -> None:
    selected = [
        {"duration_sec": 5.0, "cost_usd": 0.01, "eig_bits": 0.8, "model_uncertainty": 0.4},
        {"duration_sec": 3.0, "cost_usd": 0.006, "eig_bits": 0.5, "model_uncertainty": 0.2},
    ]
    all_tests = selected + [
        {"duration_sec": 2.0, "cost_usd": 0.004, "eig_bits": 0.2, "model_uncertainty": 0.1},
    ]
    selection_stats = {
        "budget_sec": 10.0,
        "budget_usd": 0.03,
        "decision_rationale": "protect_critical-path",
    }

    metrics = compute_metrics(selected, all_tests, selection_stats)

    assert metrics["information_possible_bits"] > metrics["bits_gained"]
    assert metrics["budget_utilization_time"] == pytest.approx(0.8)
    assert metrics["budget_utilization_cost"] == pytest.approx((0.016) / 0.03)
    assert metrics["decision_rationale"] == "protect_critical-path"


def test_emit_experiment_ledger_writes_budget_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "1234567890ab"
    selected = [
        {
            "name": "tests/test_materials.py::test_lattice_stability",
            "domain": "materials",
            "duration_sec": 5.0,
            "cost_usd": 0.01,
            "result": "fail",
            "model_uncertainty": 0.4,
            "eig_bits": 0.8,
            "entropy_before": 0.9,
            "entropy_after": 0.4,
        }
    ]
    all_tests = selected + [
        {
            "name": "tests/test_robotics.py::test_path_planning",
            "domain": "robotics",
            "duration_sec": 3.0,
            "cost_usd": 0.006,
            "result": "pass",
            "model_uncertainty": 0.2,
            "eig_bits": 0.5,
        }
    ]
    selection_stats = {"budget_sec": 10.0, "budget_usd": 0.03}
    metrics = compute_metrics(selected, all_tests, selection_stats)

    monkeypatch.setattr("scripts.gen_ci_report.get_git_sha", lambda: "f" * 40)
    monkeypatch.setattr("scripts.gen_ci_report.get_branch_name", lambda: "main")
    monkeypatch.setattr("scripts.gen_ci_report.get_env_hash", lambda: "abc123def4567890")

    ledger_dir = tmp_path / "ledger"
    emit_experiment_ledger(
        run_id=run_id,
        metrics=metrics,
        selected=selected,
        all_tests=all_tests,
        selection_stats=selection_stats,
        seed=42,
        ledger_dir=ledger_dir,
    )

    ledger_path = ledger_dir / f"{run_id}.json"
    assert ledger_path.exists()

    payload = json.loads(ledger_path.read_text())
    assert payload["budget_utilization"]["time"] == pytest.approx(metrics["budget_utilization_time"])
    assert payload["decision_rationale"] == metrics["decision_rationale"]
    assert payload["delta_entropy_bits"] == pytest.approx(metrics["delta_entropy_bits"])
    assert payload["tests"][0]["entropy_before"] == 0.9
    assert payload["tests"][0]["entropy_after"] == 0.4

