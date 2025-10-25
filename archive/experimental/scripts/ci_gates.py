#!/usr/bin/env python3
"""CI Quality Gates - Enforce coverage, calibration, and epistemic thresholds.

Reads latest experiment ledger entries, coverage reports, and calibration metrics.
Exits with code 0 if all gates pass, code 1 if any gate fails.

Usage:
    python scripts/ci_gates.py
    python scripts/ci_gates.py --coverage-file coverage.json
    python scripts/ci_gates.py --ledger experiments/ledger/
"""

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple, Any

from _config import get_thresholds, get_config


class GateResult:
    """Result of a single quality gate check."""
    
    def __init__(self, name: str, passed: bool, value: float, threshold: float, message: str = ""):
        self.name = name
        self.passed = passed
        self.value = value
        self.threshold = threshold
        self.message = message
    
    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name:20s} | {self.value:8.4f} vs {self.threshold:8.4f} | {self.message}"


def check_coverage_gate(coverage_file: pathlib.Path, min_coverage: float) -> GateResult:
    """Check coverage threshold.
    
    Args:
        coverage_file: Path to coverage.json from pytest-cov
        min_coverage: Minimum coverage percentage required
    
    Returns:
        GateResult with pass/fail status
    """
    if not coverage_file.exists():
        return GateResult(
            "Coverage",
            False,
            0.0,
            min_coverage,
            "coverage.json not found"
        )
    
    try:
        with coverage_file.open() as f:
            data = json.load(f)
        
        # pytest-cov JSON format: data["totals"]["percent_covered"]
        coverage_pct = data.get("totals", {}).get("percent_covered", 0.0)
        
        passed = coverage_pct >= min_coverage
        message = f"Coverage: {coverage_pct:.2f}%"
        
        return GateResult("Coverage", passed, coverage_pct, min_coverage, message)
    
    except Exception as e:
        return GateResult("Coverage", False, 0.0, min_coverage, f"Error: {e}")


def check_calibration_gates(ledger_dir: pathlib.Path, thresholds: Dict[str, float]) -> List[GateResult]:
    """Check calibration metrics (ECE, Brier, MCE) from latest ledger entry.
    
    Args:
        ledger_dir: Directory containing experiment ledger JSONL files
        thresholds: Dict of threshold names to values
    
    Returns:
        List of GateResults for each calibration metric
    """
    results = []
    
    # Find latest ledger file
    try:
        ledger_files = sorted(ledger_dir.glob("*.jsonl"))
        if not ledger_files:
            results.append(GateResult("ECE", False, 0.0, thresholds["ece"], "No ledger files"))
            results.append(GateResult("Brier", False, 0.0, thresholds["brier"], "No ledger files"))
            results.append(GateResult("MCE", False, 0.0, thresholds["mce"], "No ledger files"))
            return results
        
        latest_ledger = ledger_files[-1]
        
        # Read last entry (most recent)
        with latest_ledger.open() as f:
            lines = f.readlines()
            if not lines:
                results.append(GateResult("ECE", False, 0.0, thresholds["ece"], "Empty ledger"))
                results.append(GateResult("Brier", False, 0.0, thresholds["brier"], "Empty ledger"))
                results.append(GateResult("MCE", False, 0.0, thresholds["mce"], "Empty ledger"))
                return results
            
            entry = json.loads(lines[-1])
        
        # Extract calibration metrics
        calibration = entry.get("calibration", {})
        
        # Check ECE
        ece = calibration.get("ece", 0.0)
        results.append(GateResult(
            "ECE",
            ece <= thresholds["ece"],
            ece,
            thresholds["ece"],
            "Expected Calibration Error"
        ))
        
        # Check Brier Score
        brier = calibration.get("brier_score", 0.0)
        results.append(GateResult(
            "Brier",
            brier <= thresholds["brier"],
            brier,
            thresholds["brier"],
            "Brier Score"
        ))
        
        # Check MCE
        mce = calibration.get("mce", 0.0)
        results.append(GateResult(
            "MCE",
            mce <= thresholds["mce"],
            mce,
            thresholds["mce"],
            "Maximum Calibration Error"
        ))
    
    except Exception as e:
        results.append(GateResult("ECE", False, 0.0, thresholds["ece"], f"Error: {e}"))
        results.append(GateResult("Brier", False, 0.0, thresholds["brier"], f"Error: {e}"))
        results.append(GateResult("MCE", False, 0.0, thresholds["mce"], f"Error: {e}"))
    
    return results


def check_epistemic_gates(ledger_dir: pathlib.Path, thresholds: Dict[str, float]) -> List[GateResult]:
    """Check epistemic metrics (entropy, EIG, detection rate) from latest ledger.
    
    Args:
        ledger_dir: Directory containing experiment ledger JSONL files
        thresholds: Dict of threshold names to values
    
    Returns:
        List of GateResults for epistemic metrics
    """
    results = []
    
    try:
        ledger_files = sorted(ledger_dir.glob("*.jsonl"))
        if not ledger_files:
            results.append(GateResult("Entropy Delta", True, 0.0, thresholds["entropy_delta"], "No ledger (skip)"))
            results.append(GateResult("Min EIG", True, 0.0, thresholds["eig_min"], "No ledger (skip)"))
            return results
        
        latest_ledger = ledger_files[-1]
        
        with latest_ledger.open() as f:
            lines = f.readlines()
            if not lines:
                results.append(GateResult("Entropy Delta", True, 0.0, thresholds["entropy_delta"], "Empty (skip)"))
                results.append(GateResult("Min EIG", True, 0.0, thresholds["eig_min"], "Empty (skip)"))
                return results
            
            entry = json.loads(lines[-1])
        
        # Check entropy delta (should be bounded)
        uncertainty = entry.get("uncertainty", {})
        entropy_before = uncertainty.get("entropy_before", 0.0)
        entropy_after = uncertainty.get("entropy_after", 0.0)
        entropy_delta = abs(entropy_after - entropy_before)
        
        results.append(GateResult(
            "Entropy Delta",
            entropy_delta <= thresholds["entropy_delta"],
            entropy_delta,
            thresholds["entropy_delta"],
            "Information gain bounded"
        ))
        
        # Check minimum EIG (tests should provide information)
        selector_metrics = entry.get("selector_metrics", {})
        avg_eig = selector_metrics.get("avg_eig_bits", 0.0)
        
        results.append(GateResult(
            "Min EIG",
            avg_eig >= thresholds["eig_min"],
            avg_eig,
            thresholds["eig_min"],
            "Tests provide info gain"
        ))
    
    except Exception as e:
        results.append(GateResult("Entropy Delta", False, 0.0, thresholds["entropy_delta"], f"Error: {e}"))
        results.append(GateResult("Min EIG", False, 0.0, thresholds["eig_min"], f"Error: {e}"))
    
    return results


def print_gate_summary(results: List[GateResult]) -> Tuple[int, int]:
    """Print formatted gate summary and return pass/fail counts.
    
    Args:
        results: List of GateResult objects
    
    Returns:
        Tuple of (passed_count, failed_count)
    """
    print("=" * 100)
    print("CI QUALITY GATES REPORT")
    print("=" * 100)
    print()
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    for result in results:
        print(result)
    
    print()
    print("=" * 100)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 100)
    print()
    
    return passed, failed


def main() -> int:
    """Run all CI gates and return exit code.
    
    Returns:
        0 if all gates pass, 1 if any gate fails
    """
    parser = argparse.ArgumentParser(description="CI Quality Gates")
    parser.add_argument("--coverage-file", type=pathlib.Path, default="coverage.json",
                        help="Path to coverage.json (default: coverage.json)")
    parser.add_argument("--ledger-dir", type=pathlib.Path, default="experiments/ledger",
                        help="Path to experiment ledger directory (default: experiments/ledger)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on any warning (strict mode)")
    args = parser.parse_args()
    
    # Get thresholds from config
    thresholds = get_thresholds()
    config = get_config()
    
    print()
    print("=" * 100)
    print("RUNNING CI QUALITY GATES")
    print("=" * 100)
    print()
    print("Thresholds:")
    for name, value in thresholds.items():
        print(f"  {name:20s}: {value}")
    print()
    
    # Run all gates
    results: List[GateResult] = []
    
    # 1. Coverage gate
    results.append(check_coverage_gate(args.coverage_file, config["COVERAGE_MIN"]))
    
    # 2. Calibration gates
    results.extend(check_calibration_gates(args.ledger_dir, thresholds))
    
    # 3. Epistemic gates
    results.extend(check_epistemic_gates(args.ledger_dir, thresholds))
    
    # Print summary
    passed, failed = print_gate_summary(results)
    
    # Determine exit code
    if failed > 0:
        print("❌ CI GATES FAILED")
        print()
        print("To adjust thresholds, set environment variables:")
        print("  export COVERAGE_MIN=80.0")
        print("  export ECE_MAX=0.30")
        print("  export BRIER_MAX=0.25")
        print()
        return 1
    else:
        print("✅ ALL CI GATES PASSED")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
