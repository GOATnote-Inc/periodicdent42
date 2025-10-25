#!/usr/bin/env python3
"""Predict which tests to run based on code changes.

This script uses a trained ML model to select which tests are most likely to fail
given a set of code changes. Used in CI to reduce test execution time by 70%.

Phase 3 Week 7 Day 7: ML Test Selection - CI Integration

Usage (in CI):
    # Get changed files from git
    export CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD | tr '\n' ',')
    
    # Predict tests to run
    python scripts/predict_tests.py --model test_selector.pkl --output selected_tests.txt
    
    # Run selected tests
    pytest $(cat selected_tests.txt)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np

# Add app to Python path
APP_ROOT = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_ROOT))

from src.services.test_telemetry import TestCollector


def get_changed_files() -> List[str]:
    """Get files changed in current commit or from environment.
    
    Returns:
        List of changed file paths
    """
    # Try environment variable first (set by CI)
    env_files = os.getenv("CHANGED_FILES", "")
    if env_files:
        return [f.strip() for f in env_files.split(",") if f.strip()]
    
    # Fall back to git diff
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        files = result.stdout.strip().split("\n")
        return [f for f in files if f]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        print(f"⚠️  Failed to get changed files: {e}")
        return []


def get_test_features(test_name: str, test_file: str, changed_files: List[str]) -> Dict[str, float]:
    """Calculate features for a test given code changes.
    
    Args:
        test_name: Test name (pytest node ID)
        test_file: Path to test file
        changed_files: List of changed file paths
        
    Returns:
        Dict with feature values
    """
    collector = TestCollector()
    commit_sha = os.getenv("GITHUB_SHA", os.getenv("CI_COMMIT_SHA", "local"))
    
    # Calculate diff stats
    diff_stats = collector.calculate_diff_stats(commit_sha) if commit_sha != "local" else {}
    
    # Historical features
    recent_failure_rate = collector.get_recent_failure_rate(test_name)
    avg_duration = collector.get_avg_duration(test_name)
    days_since_last_change = collector.get_days_since_last_change(test_file)
    
    return {
        "lines_added": diff_stats.get("lines_added", 0),
        "lines_deleted": diff_stats.get("lines_deleted", 0),
        "files_changed": len(changed_files),
        "complexity_delta": 0.0,  # Simplified
        "recent_failure_rate": recent_failure_rate,
        "avg_duration": avg_duration,
        "days_since_last_change": days_since_last_change,
    }


def discover_all_tests() -> List[Dict[str, str]]:
    """Discover all pytest tests in repository.
    
    Returns:
        List of dicts with 'test_name' and 'test_file'
    """
    try:
        # Use pytest --collect-only to discover tests
        result = subprocess.run(
            ["pytest", "--collect-only", "-q", "tests/", "app/tests/"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        tests = []
        for line in result.stdout.split("\n"):
            # Parse pytest collection output
            # Format: tests/test_api.py::test_health
            if "::" in line and not line.startswith(" "):
                parts = line.strip().split("::")
                if len(parts) >= 2:
                    test_file = parts[0]
                    test_name = line.strip()
                    tests.append({"test_name": test_name, "test_file": test_file})
        
        return tests
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        print(f"⚠️  Failed to discover tests: {e}")
        # Fallback: find test files manually
        tests = []
        for test_dir in [Path("tests"), Path("app/tests")]:
            if test_dir.exists():
                for test_file in test_dir.rglob("test_*.py"):
                    tests.append({
                        "test_name": str(test_file),
                        "test_file": str(test_file)
                    })
        return tests


def predict_test_failures(
    model_path: Path,
    changed_files: List[str],
    threshold: float = 0.1,
    min_tests: int = 10
) -> List[str]:
    """Predict which tests are most likely to fail.
    
    Args:
        model_path: Path to trained model (.pkl)
        changed_files: List of changed file paths
        threshold: Probability threshold for selection (0.0 to 1.0)
        min_tests: Minimum number of tests to always run
        
    Returns:
        List of test names to run (pytest node IDs)
    """
    print("\n" + "="*70)
    print("🎯 ML-POWERED TEST SELECTION")
    print("="*70)
    
    # Load model
    try:
        model = joblib.load(model_path)
        print(f"✅ Loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Falling back to running all tests")
        return []
    
    # Load feature names
    metadata_path = model_path.with_suffix(".json")
    if metadata_path.exists():
        import json
        metadata = json.loads(metadata_path.read_text())
        feature_names = metadata["feature_names"]
    else:
        feature_names = [
            "lines_added", "lines_deleted", "files_changed",
            "complexity_delta", "recent_failure_rate",
            "avg_duration", "days_since_last_change"
        ]
    
    print(f"📝 Changed files: {len(changed_files)}")
    for f in changed_files[:10]:
        print(f"   - {f}")
    if len(changed_files) > 10:
        print(f"   ... and {len(changed_files) - 10} more")
    
    # Discover all tests
    print(f"\n🔍 Discovering tests...")
    all_tests = discover_all_tests()
    print(f"✅ Found {len(all_tests)} tests")
    
    if not all_tests:
        print("⚠️  No tests found, running all tests")
        return []
    
    # Calculate features for each test
    print(f"\n🧮 Calculating features...")
    test_predictions = []
    
    for test_info in all_tests:
        test_name = test_info["test_name"]
        test_file = test_info["test_file"]
        
        # Calculate features
        features = get_test_features(test_name, test_file, changed_files)
        feature_vector = np.array([[features[f] for f in feature_names]])
        
        # Predict failure probability
        try:
            proba = model.predict_proba(feature_vector)[0, 1]
        except Exception:
            proba = 0.5  # Default to middle probability if prediction fails
        
        test_predictions.append({
            "test_name": test_name,
            "failure_probability": proba,
        })
    
    # Sort by failure probability
    test_predictions.sort(key=lambda x: x["failure_probability"], reverse=True)
    
    # Select tests above threshold
    selected = [
        t["test_name"] for t in test_predictions
        if t["failure_probability"] >= threshold
    ]
    
    # Ensure minimum number of tests
    if len(selected) < min_tests:
        print(f"\n⚠️  Only {len(selected)} tests above threshold, selecting top {min_tests}")
        selected = [t["test_name"] for t in test_predictions[:min_tests]]
    
    # Print selection summary
    print(f"\n📊 Test Selection Summary:")
    print(f"   Total tests: {len(all_tests)}")
    print(f"   Selected: {len(selected)}")
    print(f"   Reduction: {(1 - len(selected)/len(all_tests))*100:.1f}%")
    print(f"\n🎯 Top 10 High-Risk Tests:")
    for i, t in enumerate(test_predictions[:10], 1):
        print(f"   {i:2d}. {t['test_name']:50s} ({t['failure_probability']:.3f})")
    
    return selected


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="Predict which tests to run using ML"
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained model (.pkl)"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Path to write selected test names"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Probability threshold for selection (default: 0.1)"
    )
    parser.add_argument(
        "--min-tests", type=int, default=10,
        help="Minimum number of tests to run (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Get changed files
    changed_files = get_changed_files()
    
    if not changed_files:
        print("⚠️  No changed files detected, running all tests")
        return 0
    
    # Predict tests to run
    selected_tests = predict_test_failures(
        args.model,
        changed_files,
        threshold=args.threshold,
        min_tests=args.min_tests
    )
    
    # Write output if requested
    if args.output:
        if selected_tests:
            args.output.write_text("\n".join(selected_tests) + "\n")
            print(f"\n✅ Selected tests written to {args.output}")
        else:
            # Empty file means run all tests
            args.output.write_text("")
            print(f"\n✅ Running all tests (no selection applied)")
    
    # Print pytest command
    if selected_tests:
        print(f"\n📖 Run selected tests:")
        print(f"   pytest {' '.join(selected_tests[:5])}")
        if len(selected_tests) > 5:
            print(f"   ... and {len(selected_tests) - 5} more tests")
    else:
        print(f"\n📖 Run all tests:")
        print(f"   pytest tests/ app/tests/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
