#!/usr/bin/env python3
"""Collect CI run metadata with epistemic features.

Supports both real CI data (from env vars) and synthetic mock data
for testing the epistemic CI pipeline.

Usage:
    # Real CI run
    python scripts/collect_ci_runs.py
    
    # Mock data with controlled failure rate (deterministic with seed)
    python scripts/collect_ci_runs.py --mock 100 --inject-failures 0.12 --seed 42
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


# Domain-specific test suites
DOMAIN_TESTS = {
    "materials": [
        "tests/test_materials.py::test_lattice_stability",
        "tests/test_materials.py::test_dft_convergence",
        "tests/test_materials.py::test_phonon_dispersion",
        "tests/test_materials.py::test_elastic_constants",
        "tests/test_materials.py::test_band_structure",
        "tests/test_materials.py::test_dos_integration",
        "tests/test_materials.py::test_surface_energy",
        "tests/test_materials.py::test_defect_formation",
    ],
    "protein": [
        "tests/test_protein.py::test_folding_energy",
        "tests/test_protein.py::test_binding_affinity",
        "tests/test_protein.py::test_stability_score",
        "tests/test_protein.py::test_solubility",
        "tests/test_protein.py::test_sequence_alignment",
        "tests/test_protein.py::test_secondary_structure",
        "tests/test_protein.py::test_hydrophobicity",
        "tests/test_protein.py::test_mutation_effects",
    ],
    "robotics": [
        "tests/test_robotics.py::test_inverse_kinematics",
        "tests/test_robotics.py::test_path_planning",
        "tests/test_robotics.py::test_collision_detection",
        "tests/test_robotics.py::test_control_loop",
        "tests/test_robotics.py::test_sensor_fusion",
        "tests/test_robotics.py::test_trajectory_optimization",
        "tests/test_robotics.py::test_gripper_force",
        "tests/test_robotics.py::test_localization",
    ],
    "generic": [
        "tests/test_health.py::test_health_endpoint",
        "tests/test_reasoning_smoke.py::test_reasoning_endpoint",
        "tests/test_phase2_scientific.py::test_numerical_precision",
        "tests/chaos/test_chaos_examples.py::test_resilient_api",
    ],
}

FAILURE_TYPES = ["assertion", "timeout", "exception", "numerical", "flaky"]


def generate_mock_test(
    name: str,
    domain: str,
    failure_prob: float = 0.1,
    runner_usd_per_hour: float = 0.60
) -> Dict[str, Any]:
    """Generate a single mock test result.
    
    Args:
        name: Test name
        domain: Test domain (materials|protein|robotics|generic)
        failure_prob: Probability of test failure
        runner_usd_per_hour: CI runner cost per hour
        
    Returns:
        Test dict matching schema
    """
    # Simulate test execution
    duration = random.uniform(0.5, 30.0)  # 0.5s to 30s
    
    # Decide pass/fail
    result = "fail" if random.random() < failure_prob else "pass"
    
    # Generate model uncertainty (for ML-based prediction)
    # High uncertainty near p=0.5, low near 0 or 1
    if result == "fail":
        # Failed tests should have higher predicted failure prob
        model_uncertainty = random.uniform(0.4, 0.9)
    else:
        # Passing tests should have lower predicted failure prob
        model_uncertainty = random.uniform(0.05, 0.4)
    
    test = {
        "name": name,
        "suite": domain,
        "domain": domain,
        "duration_sec": duration,
        "result": result,
        "cost_usd": duration * runner_usd_per_hour / 3600.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_uncertainty": model_uncertainty,
    }
    
    # Add failure_type if failed
    if result == "fail":
        test["failure_type"] = random.choice(FAILURE_TYPES)
    
    # Add domain-specific metrics
    if domain == "materials":
        test["metrics"] = {
            "convergence_error": random.uniform(1e-6, 1e-3) if result == "pass" else random.uniform(1e-3, 1e-1),
            "lattice_deviation": random.uniform(0.001, 0.01),
        }
    elif domain == "protein":
        test["metrics"] = {
            "binding_affinity_kcal_mol": random.uniform(-15, -5),
            "folding_stability": random.uniform(0.7, 0.99),
        }
    elif domain == "robotics":
        test["metrics"] = {
            "trajectory_error_mm": random.uniform(0.1, 5.0),
            "control_latency_ms": random.uniform(1.0, 10.0),
        }
    
    return test


def generate_mock_run(
    n_tests: int,
    failure_prob: float = 0.1,
    runner_usd_per_hour: float = 0.60
) -> Dict[str, Any]:
    """Generate a mock CI run with N tests.
    
    Args:
        n_tests: Number of tests to generate
        failure_prob: Probability of test failure
        runner_usd_per_hour: CI runner cost per hour
        
    Returns:
        CIRun dict matching schema
    """
    # Generate diverse test mix
    all_tests = []
    for domain, tests in DOMAIN_TESTS.items():
        domain_count = max(1, n_tests // 4)  # Distribute evenly
        selected = random.choices(tests, k=domain_count)
        for test_name in selected:
            all_tests.append(generate_mock_test(
                test_name,
                domain,
                failure_prob,
                runner_usd_per_hour
            ))
    
    # Trim to exactly n_tests
    all_tests = all_tests[:n_tests]
    
    # Compute CI run metadata
    walltime = sum(t["duration_sec"] for t in all_tests)
    total_cost = sum(t["cost_usd"] for t in all_tests)
    
    run = {
        "commit": f"mock-{random.randint(100000, 999999):06x}",
        "branch": random.choice(["main", "feature/epistemic-ci", "fix/test-selection"]),
        "changed_files": [
            f"src/{random.choice(['materials', 'protein', 'robotics'])}/{random.choice(['model', 'optimizer', 'validator'])}.py"
            for _ in range(random.randint(1, 5))
        ],
        "lines_added": random.randint(10, 500),
        "lines_deleted": random.randint(5, 200),
        "walltime_sec": walltime,
        "cpu_sec": walltime * random.uniform(0.9, 1.1),  # Slightly variable
        "mem_gb_sec": walltime * random.uniform(0.5, 2.0),  # Memory usage
        "tests": all_tests,
        "budget_sec": walltime * 0.5,  # Default budget: 50% of full suite
        "budget_usd": total_cost * 0.5,
        "runner_usd_per_hour": runner_usd_per_hour,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return run


def collect_real_run(runner_usd_per_hour: float = 0.60) -> Dict[str, Any]:
    """Collect real CI run data from environment variables.
    
    Args:
        runner_usd_per_hour: CI runner cost per hour
        
    Returns:
        CIRun dict (minimal, to be enriched by pytest plugin)
    """
    # Basic run metadata from GitHub Actions env
    run = {
        "commit": os.getenv("GITHUB_SHA", "unknown"),
        "branch": os.getenv("GITHUB_REF_NAME", "unknown"),
        "changed_files": [],  # To be filled by git diff
        "lines_added": 0,
        "lines_deleted": 0,
        "walltime_sec": float(os.getenv("CI_DURATION_SEC", "0") or "0"),
        "tests": [],  # To be filled by pytest telemetry
        "budget_sec": 0,  # To be computed
        "runner_usd_per_hour": runner_usd_per_hour,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return run


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_env_hash() -> str:
    """Compute hash of relevant environment variables for reproducibility."""
    env_keys = sorted([
        "PYTHON_VERSION",
        "PYTEST_VERSION",
        "RUNNER_USD_PER_HOUR",
        "CI_BUDGET_FRACTION",
    ])
    env_str = "|".join(f"{k}={os.getenv(k, '')}" for k in env_keys)
    return hashlib.sha256(env_str.encode()).hexdigest()[:16]


def emit_run_metadata(seed: Optional[int], output_dir: pathlib.Path) -> None:
    """Emit reproducibility metadata to artifact/run_meta.json.
    
    Args:
        seed: Random seed used (None if not set)
        output_dir: Directory for metadata artifact
    """
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "env_hash": get_env_hash(),
        "seed": seed,
        "script": "collect_ci_runs.py",
        "reproducible": seed is not None,
    }
    
    artifact_dir = output_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = artifact_dir / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📝 Emitted metadata to {meta_path}", flush=True)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect CI run telemetry for epistemic test selection"
    )
    parser.add_argument(
        "--mock",
        type=int,
        metavar="N",
        help="Generate N mock tests instead of real CI data"
    )
    parser.add_argument(
        "--inject-failures",
        type=float,
        metavar="P",
        default=0.1,
        help="Failure probability for mock data (default: 0.1)"
    )
    parser.add_argument(
        "--out",
        default="data/ci_runs.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--runner-cost",
        type=float,
        default=0.60,
        help="CI runner cost per hour (USD)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="Random seed for reproducibility (default: random)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}", flush=True)
    
    output_path = pathlib.Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Emit reproducibility metadata
    emit_run_metadata(args.seed, output_path.parent.parent if output_path.parent.name == "data" else output_path.parent)
    
    # Generate or collect run
    if args.mock:
        print(f"🎭 Generating {args.mock} mock tests (failure rate: {args.inject_failures:.1%})", flush=True)
        run = generate_mock_run(args.mock, args.inject_failures, args.runner_cost)
    else:
        print("📊 Collecting real CI run data", flush=True)
        run = collect_real_run(args.runner_cost)
    
    # Append to JSONL
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")
    
    # Report
    n_tests = len(run.get("tests", []))
    if n_tests > 0:
        n_failures = sum(1 for t in run["tests"] if t["result"] == "fail")
        total_time = sum(t["duration_sec"] for t in run["tests"])
        total_cost = sum(t["cost_usd"] for t in run["tests"])
        
        print(f"✅ Collected CI run:", flush=True)
        print(f"   Tests: {n_tests} ({n_failures} failed)", flush=True)
        print(f"   Time: {total_time:.1f}s", flush=True)
        print(f"   Cost: ${total_cost:.4f}", flush=True)
        print(f"   Commit: {run['commit']}", flush=True)
    else:
        print(f"✅ Recorded run metadata (tests to be added by pytest)", flush=True)
    
    print(f"📝 Wrote to {output_path}", flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())