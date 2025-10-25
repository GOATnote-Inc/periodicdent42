from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

from synthloop.orchestrator.main import app, run_manager
from synthloop.orchestrator.storage import db
from synthloop.orchestrator.models.plan import SynthesisPlan


@pytest.fixture(autouse=True)
def reset_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("EVENT_LOG_DIR", str(tmp_path / "events"))
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BASIC_AUTH_USER", "test")
    monkeypatch.setenv("BASIC_AUTH_PASS", "test")
    monkeypatch.setenv("MAX_T_C", "680")
    monkeypatch.setenv("MAX_RAMP_C_PER_MIN", "10")
    monkeypatch.setenv("MAX_RPM", "600")
    monkeypatch.setenv("MAX_BATCH_G", "20")
    monkeypatch.setenv("WATCHDOG_TIMEOUT_S", "1")
    db.reset_engine()
    run_manager.runs.clear()
    client = TestClient(app)
    yield
    client.close()


def auth_client():
    client = TestClient(app)
    client.auth = ("test", "test")
    return client


def create_plan(tolerance=0.1, heat_target=650, ramp=5):
    plan = {
        "plan_id": "test",
        "operator": "tester",
        "operator_ack": True,
        "backend": "sim",
        "sample": {"id": "S1", "batch_g": 5.0, "tolerance_g": tolerance},
        "reagents": [
            {"name": "A", "lot": "A", "target_g": 3.0},
            {"name": "B", "lot": "B", "target_g": 2.0},
        ],
        "steps": [
            {"type": "Dispense", "reagent": "A", "mass_g": 3.0},
            {"type": "Dispense", "reagent": "B", "mass_g": 2.0},
            {"type": "Mix", "duration_s": 10, "rpm": 200},
            {"type": "Heat", "target_C": heat_target, "ramp_C_per_min": ramp, "hold_min": 0.5},
            {"type": "Cool", "target_C": 25},
            {"type": "Collect", "sample_id": "S1"},
        ],
        "calibration_refs": {
            "scale_id": "SCALE",
            "temp_probe_id": "TMP",
            "last_calibrated": datetime.utcnow().isoformat(),
            "last_calibrated_temp": datetime.utcnow().isoformat(),
        },
    }
    return plan


def submit_and_start(client, plan):
    res = client.post("/synthesis/plans", json={"plan": plan}, auth=client.auth)
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    res = client.post(f"/synthesis/runs/{run_id}/start", auth=client.auth)
    assert res.status_code == 200
    return run_id


def wait_for_completion(client, run_id, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/synthesis/runs/{run_id}", auth=client.auth)
        assert res.status_code == 200
        data = res.json()
        if data["status"] in {"completed", "failed", "aborted"} and data.get("outcome"):
            return data
        time.sleep(0.5)
    raise AssertionError("timeout waiting for run")


def test_interlock_violation(monkeypatch):
    client = auth_client()
    plan = create_plan(heat_target=700)
    monkeypatch.setenv("MAX_T_C", "650")
    res = client.post("/synthesis/plans", json={"plan": plan}, auth=client.auth)
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    client.post(f"/synthesis/runs/{run_id}/start", auth=client.auth)
    data = wait_for_completion(client, run_id)
    assert data["status"] in {"failed"}
    assert data["outcome"]["failure_mode"] == "InterlockViolation"


def test_abort_during_heat(monkeypatch):
    client = auth_client()
    plan = create_plan(heat_target=640)
    run_id = submit_and_start(client, plan)
    time.sleep(1)
    res = client.post(f"/synthesis/runs/{run_id}/abort", auth=client.auth)
    assert res.status_code == 200
    data = wait_for_completion(client, run_id)
    assert data["status"] == "aborted"
    assert data["outcome"]["failure_mode"] == "Abort"


def test_qc_fail_due_to_mass_balance():
    client = auth_client()
    plan = create_plan(tolerance=0.0001)
    run_id = submit_and_start(client, plan)
    data = wait_for_completion(client, run_id)
    assert data["status"] == "failed"
    assert data["outcome"]["failure_mode"] == "QCFail"


def test_outcome_persistence():
    client = auth_client()
    plan = create_plan()
    run_id = submit_and_start(client, plan)
    data = wait_for_completion(client, run_id)
    res = client.get("/synthesis/runs", params={"outcome": "negative"}, auth=client.auth)
    assert res.status_code == 200
    runs = res.json()
    assert any(r["run_id"] == run_id for r in runs) == (not data["outcome"]["success"])


