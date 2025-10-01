from __future__ import annotations

import os
import time
import uuid

import requests
from yaml import safe_load

API = os.environ.get("SYNTH_API", "http://localhost:8080")
AUTH = (os.environ.get("BASIC_AUTH_USER", "admin"), os.environ.get("BASIC_AUTH_PASS", "changeme"))


def load_plan(name: str):
    with open(os.path.join(os.path.dirname(__file__), "..", "examples", name), "r", encoding="utf-8") as fh:
        return safe_load(fh)


def submit(plan):
    res = requests.post(f"{API}/synthesis/plans", json={"plan": plan}, auth=AUTH)
    res.raise_for_status()
    return res.json()["run_id"]


def start(run_id):
    res = requests.post(f"{API}/synthesis/runs/{run_id}/start", auth=AUTH)
    res.raise_for_status()


def wait(run_id):
    while True:
        res = requests.get(f"{API}/synthesis/runs/{run_id}", auth=AUTH)
        res.raise_for_status()
        data = res.json()
        if data["status"] in {"completed", "failed", "aborted"} and data.get("outcome"):
            print(f"Run {run_id} -> {data['status']} ({data['outcome']['failure_mode']})")
            break
        time.sleep(1)


def main():
    plans = [
        ("powder_plan_ok.yaml", {}),
        ("powder_plan_ok.yaml", {"sample": {"tolerance_g": 0.0001}}),
        ("powder_plan_fail_overtemp.yaml", {}),
    ]
    for idx, (name, overrides) in enumerate(plans, 1):
        plan = load_plan(name)
        for path, value in overrides.items():
            keys = path.split(".")
            target = plan
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = value
        if idx == 3:
            os.environ["SIM_FAULT_MODE"] = "heater_overshoot"
        else:
            os.environ["SIM_FAULT_MODE"] = "none"
        run_id = submit(plan)
        start(run_id)
        wait(run_id)


if __name__ == "__main__":
    main()

