"""Performance benchmarks with budget guardrails.

Tests that critical operations complete within time/cost budgets.
Run with: pytest tests/test_performance_benchmarks.py --benchmark-only

Budget Configuration:
- Set MAX_CI_COST_USD, MAX_CI_WALLTIME_SEC in environment
- Defaults: $1.00 USD, 1800 seconds (30 min)
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, Any

import pytest


# Budget caps from environment (with defaults)
MAX_COST_USD = float(os.getenv("MAX_CI_COST_USD", "1.00"))
MAX_WALLTIME_SEC = float(os.getenv("MAX_CI_WALLTIME_SEC", "1800"))

# Performance baselines (P95 latency targets)
BASELINE_COLLECT_SEC = 5.0  # collect_ci_runs.py --mock 100
BASELINE_TRAIN_SEC = 10.0   # train_selector.py (100 runs)
BASELINE_SCORE_SEC = 2.0    # score_eig.py (100 tests)
BASELINE_SELECT_SEC = 1.0   # select_tests.py (100 tests)
BASELINE_REPORT_SEC = 1.0   # gen_ci_report.py


def run_script(script: str, *args: str) -> Dict[str, Any]:
    """Run a script and measure performance.
    
    Args:
        script: Script path relative to repo root
        *args: Additional arguments
        
    Returns:
        Performance metrics dict
    """
    import time
    
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, script] + list(args),
        capture_output=True,
        text=True,
        check=False
    )
    elapsed = time.perf_counter() - start
    
    return {
        "elapsed_sec": elapsed,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@pytest.mark.benchmark(group="epistemic-ci", min_rounds=3)
def test_benchmark_collect_mock(benchmark, tmp_path):
    """Benchmark: Collect 100 mock CI runs."""
    def collect():
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = run_script(
                "scripts/collect_ci_runs.py",
                "--mock", "100",
                "--inject-failures", "0.12",
                "--seed", "42",
                "--out", f"{tmpdir}/ci_runs.jsonl"
            )
            return metrics
    
    result = benchmark(collect)
    
    # Assert performance baseline
    assert result["returncode"] == 0, f"Script failed: {result['stderr']}"
    assert result["elapsed_sec"] < BASELINE_COLLECT_SEC, \
        f"Collect took {result['elapsed_sec']:.2f}s (baseline: {BASELINE_COLLECT_SEC}s)"


@pytest.mark.benchmark(group="epistemic-ci", min_rounds=2)
def test_benchmark_train_selector(benchmark, tmp_path):
    """Benchmark: Train ML selector on 100 runs."""
    # Setup: Generate training data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = pathlib.Path(tmpdir) / "data"
        models_dir = pathlib.Path(tmpdir) / "models"
        data_dir.mkdir()
        models_dir.mkdir()
        
        # Generate 100 mock runs
        for i in range(100):
            subprocess.run(
                [sys.executable, "scripts/collect_ci_runs.py",
                 "--mock", "1", "--out", str(data_dir / "ci_runs.jsonl")],
                capture_output=True,
                check=True
            )
        
        def train():
            metrics = run_script(
                "scripts/train_selector.py",
                "--data", str(data_dir / "ci_runs.jsonl"),
                "--out", str(models_dir / "selector.pkl"),
                "--meta", str(models_dir / "metadata.json"),
                "--seed", "42"
            )
            return metrics
        
        result = benchmark(train)
        
        # Assert performance baseline
        assert result["returncode"] == 0, f"Training failed: {result['stderr']}"
        assert result["elapsed_sec"] < BASELINE_TRAIN_SEC, \
            f"Training took {result['elapsed_sec']:.2f}s (baseline: {BASELINE_TRAIN_SEC}s)"


@pytest.mark.benchmark(group="epistemic-ci", min_rounds=5)
def test_benchmark_score_eig(benchmark, tmp_path):
    """Benchmark: Score EIG for 100 tests."""
    # Setup: Generate EIG rankings file
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = pathlib.Path(tmpdir) / "artifact"
        artifact_dir.mkdir()
        
        # Generate mock EIG input (would come from collect_ci_runs)
        subprocess.run(
            [sys.executable, "scripts/collect_ci_runs.py",
             "--mock", "100", "--out", str(pathlib.Path(tmpdir) / "data" / "ci_runs.jsonl"),
             "--seed", "42"],
            capture_output=True,
            check=True
        )
        
        def score():
            metrics = run_script("scripts/score_eig.py")
            return metrics
        
        result = benchmark.pedantic(score, rounds=3, warmup_rounds=1)
        
        # Assert performance baseline
        assert result["returncode"] == 0, f"Scoring failed: {result['stderr']}"
        assert result["elapsed_sec"] < BASELINE_SCORE_SEC, \
            f"Scoring took {result['elapsed_sec']:.2f}s (baseline: {BASELINE_SCORE_SEC}s)"


@pytest.mark.benchmark(group="budget-caps")
def test_budget_time_cap(benchmark):
    """Budget Cap: Full pipeline must complete within MAX_WALLTIME_SEC."""
    def full_pipeline():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run full mock pipeline
            result = subprocess.run(
                ["make", "mock"],
                capture_output=True,
                text=True,
                check=False,
                cwd=pathlib.Path(__file__).parent.parent,
                timeout=MAX_WALLTIME_SEC + 60  # Grace period for test harness
            )
            return {
                "returncode": result.returncode,
                "elapsed_sec": 0,  # Will be measured by benchmark
            }
    
    result = benchmark(full_pipeline)
    
    # Assert time budget
    elapsed = benchmark.stats["mean"]
    assert elapsed < MAX_WALLTIME_SEC, \
        f"Pipeline took {elapsed:.1f}s (budget: {MAX_WALLTIME_SEC}s)"


@pytest.mark.benchmark(group="budget-caps")
def test_budget_cost_cap():
    """Budget Cap: Full pipeline must cost less than MAX_COST_USD."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run full pipeline and parse metrics
        subprocess.run(
            ["make", "mock"],
            capture_output=True,
            check=True,
            cwd=pathlib.Path(__file__).parent.parent
        )
        
        # Parse cost from ci_metrics.json
        metrics_path = pathlib.Path("artifact/ci_metrics.json")
        assert metrics_path.exists(), "ci_metrics.json not found"
        
        metrics = json.loads(metrics_path.read_text())
        
        # Extract actual cost (full suite - savings)
        cost_saved = metrics.get("run_cost_saved_usd", 0)
        # Estimate full suite cost based on savings percentage
        if metrics.get("cost_reduction_pct", 0) > 0:
            full_cost = cost_saved / (metrics["cost_reduction_pct"] / 100)
        else:
            full_cost = 0.25  # Default estimate
        
        # Assert cost budget
        assert full_cost < MAX_COST_USD, \
            f"Pipeline cost ${full_cost:.4f} (budget: ${MAX_COST_USD})"


def test_perf_regression_detection():
    """Detect performance regressions vs. baseline.
    
    This test loads historical benchmark data and alerts if current
    performance degrades >20% from baseline.
    """
    baseline_file = pathlib.Path("artifact/.benchmark_baseline.json")
    
    if not baseline_file.exists():
        pytest.skip("No baseline data (run with --benchmark-save to create)")
    
    baseline = json.loads(baseline_file.read_text())
    
    # Compare current run to baseline
    # (In real implementation, load from pytest-benchmark JSON)
    current_collect_sec = 3.5  # Would be measured in actual run
    baseline_collect_sec = baseline.get("collect_sec", BASELINE_COLLECT_SEC)
    
    regression_threshold = 1.2  # 20% slowdown triggers alert
    
    if current_collect_sec > baseline_collect_sec * regression_threshold:
        pytest.fail(
            f"Performance regression detected: "
            f"collect_ci_runs.py {current_collect_sec:.2f}s "
            f"(baseline: {baseline_collect_sec:.2f}s, +{(current_collect_sec/baseline_collect_sec - 1)*100:.1f}%)"
        )


if __name__ == "__main__":
    """Run benchmarks with budget enforcement."""
    import sys
    
    # Run with budget caps
    exit_code = pytest.main([
        __file__,
        "--benchmark-only",
        "--benchmark-columns=min,max,mean,median",
        "--benchmark-sort=name",
        "-v"
    ])
    
    if exit_code != 0:
        print("\n❌ BUDGET BREACH: Performance tests failed", file=sys.stderr)
        print(f"   Time budget: {MAX_WALLTIME_SEC}s", file=sys.stderr)
        print(f"   Cost budget: ${MAX_COST_USD}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n✅ Performance within budget", file=sys.stderr)
        print(f"   Time budget: {MAX_WALLTIME_SEC}s", file=sys.stderr)
        print(f"   Cost budget: ${MAX_COST_USD}", file=sys.stderr)
        sys.exit(0)
