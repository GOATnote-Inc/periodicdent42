#!/usr/bin/env python3
"""Collect test telemetry data by running tests multiple times.

This script runs the test suite N times to collect real execution data
for training the ML test selection model.

Phase 3 Validation - Day 1-2: Real CI Data Collection

Usage:
    python scripts/collect_test_telemetry.py --runs 50 --fast
    python scripts/collect_test_telemetry.py --runs 100 --tests tests/test_health.py
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import os

# Add app to path
APP_ROOT = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_ROOT))

try:
    from src.services.test_telemetry import TestCollector
    import psycopg2
except ImportError as e:
    print(f"❌ Failed to import dependencies: {e}")
    print("💡 Run: cd app && pip install -e .")
    sys.exit(1)


def get_db_connection():
    """Get database connection to check telemetry data."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        user=os.getenv("DB_USER", "ard_user"),
        password=os.getenv("DB_PASSWORD", "ard_secure_password_2024"),
        dbname=os.getenv("DB_NAME", "ard_intelligence"),
    )


def count_telemetry_records():
    """Count current telemetry records in database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_telemetry")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"⚠️  Could not check database: {e}")
        return -1


def run_tests(test_path: str = "tests/", pytest_args: str = "") -> tuple[int, float]:
    """Run pytest and collect telemetry.
    
    Args:
        test_path: Path to tests to run
        pytest_args: Additional pytest arguments
        
    Returns:
        (exit_code, duration_seconds)
    """
    # Set environment for telemetry collection
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["DB_HOST"] = "localhost"
    env["DB_PORT"] = "5433"
    env["DB_USER"] = "ard_user"
    env["DB_PASSWORD"] = "ard_secure_password_2024"
    env["DB_NAME"] = "ard_intelligence"
    
    # Don't skip telemetry
    env.pop("SKIP_TEST_TELEMETRY", None)
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        "-m", "not chem and not slow",  # Fast tests only
    ]
    
    if pytest_args:
        cmd.extend(pytest_args.split())
    
    # Run tests
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.time() - start
    
    return result.returncode, duration


def main():
    parser = argparse.ArgumentParser(description="Collect test telemetry data")
    parser.add_argument("--runs", type=int, default=50,
                       help="Number of test runs (default: 50)")
    parser.add_argument("--tests", type=str, default="tests/",
                       help="Test path to run (default: tests/)")
    parser.add_argument("--fast", action="store_true",
                       help="Run only fast tests (skip chem, slow)")
    parser.add_argument("--pytest-args", type=str, default="",
                       help="Additional pytest arguments")
    
    args = parser.parse_args()
    
    print("━" * 80)
    print("🔬 Test Telemetry Data Collection")
    print("━" * 80)
    print(f"Runs to execute: {args.runs}")
    print(f"Test path: {args.tests}")
    print(f"Fast mode: {args.fast}")
    print()
    
    # Check initial database state
    initial_count = count_telemetry_records()
    if initial_count == -1:
        print("❌ Database connection failed. Check Cloud SQL Proxy.")
        return 1
    
    print(f"📊 Current telemetry records: {initial_count}")
    print()
    
    # Run tests N times
    start_time = datetime.now()
    successful_runs = 0
    failed_runs = 0
    total_duration = 0.0
    
    for i in range(1, args.runs + 1):
        print(f"🧪 Run {i}/{args.runs}...", end=" ", flush=True)
        
        try:
            exit_code, duration = run_tests(args.tests, args.pytest_args)
            total_duration += duration
            
            if exit_code == 0:
                print(f"✅ PASS ({duration:.1f}s)")
                successful_runs += 1
            else:
                print(f"⚠️  FAIL ({duration:.1f}s)")
                failed_runs += 1
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed_runs += 1
        
        # Show progress every 10 runs
        if i % 10 == 0:
            current_count = count_telemetry_records()
            new_records = current_count - initial_count if current_count >= 0 else "?"
            avg_duration = total_duration / i
            print(f"   📈 Progress: {new_records} new records, avg {avg_duration:.1f}s/run")
            print()
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    final_count = count_telemetry_records()
    new_records = final_count - initial_count if final_count >= 0 else "?"
    
    print()
    print("━" * 80)
    print("✅ Data Collection Complete")
    print("━" * 80)
    print(f"Total runs: {args.runs}")
    print(f"Successful: {successful_runs} ({successful_runs/args.runs*100:.1f}%)")
    print(f"Failed: {failed_runs} ({failed_runs/args.runs*100:.1f}%)")
    print(f"Total duration: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Avg per run: {total_duration/args.runs:.1f}s")
    print()
    print(f"📊 Telemetry records collected: {new_records}")
    print(f"   Before: {initial_count}")
    print(f"   After: {final_count}")
    print()
    
    if new_records > 0:
        print("✅ Ready to retrain ML model!")
        print("   Next step: python scripts/train_test_selector.py --real-data")
    else:
        print("⚠️  No new telemetry records collected.")
        print("   Check that telemetry plugin is working:")
        print("   - DEBUG_TELEMETRY=1 pytest tests/test_health.py -v")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
