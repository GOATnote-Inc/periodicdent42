#!/usr/bin/env python3
"""Flaky Test Scanner - Detect tests with inconsistent pass/fail behavior.

Parses JUnit XML reports and tracks flip count over last K runs.
Marks tests with >2 flips in last 10 runs as flaky.

Output: evidence/regressions/flaky_tests.json
"""

import argparse
import json
import pathlib
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _config import get_config


def parse_junit_xml(xml_file: pathlib.Path) -> Dict[str, str]:
    """Parse JUnit XML and extract test results.
    
    Args:
        xml_file: Path to JUnit XML file
    
    Returns:
        Dict of {test_name: status} where status in ('pass', 'fail', 'skip', 'error')
    """
    results = {}
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for testcase in root.iter('testcase'):
            classname = testcase.get('classname', '')
            name = testcase.get('name', '')
            test_name = f"{classname}::{name}" if classname else name
            
            # Determine status
            if testcase.find('failure') is not None:
                status = 'fail'
            elif testcase.find('error') is not None:
                status = 'error'
            elif testcase.find('skipped') is not None:
                status = 'skip'
            else:
                status = 'pass'
            
            results[test_name] = status
    
    except Exception:
        pass
    
    return results


def compute_flip_count(test_history: List[str]) -> int:
    """Compute number of status flips in test history.
    
    Args:
        test_history: List of statuses (oldest first)
    
    Returns:
        Number of flips
    """
    if len(test_history) < 2:
        return 0
    
    flips = 0
    for i in range(1, len(test_history)):
        if test_history[i] != test_history[i-1]:
            flips += 1
    
    return flips


def scan_flaky_tests(tests_dir: pathlib.Path, window: int = 10, flip_threshold: int = 2) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Scan for flaky tests in recent JUnit XMLs.
    
    Args:
        tests_dir: Directory containing JUnit XML files
        window: Number of runs to analyze
        flip_threshold: Number of flips to mark as flaky
    
    Returns:
        Tuple of (summary dict, list of flaky test dicts)
    """
    test_history = defaultdict(list)
    
    # Load JUnit XMLs (sorted by filename, assuming chronological)
    xml_files = sorted(tests_dir.glob("*.xml"))[-window:]
    
    for xml_file in xml_files:
        results = parse_junit_xml(xml_file)
        for test_name, status in results.items():
            test_history[test_name].append(status)
    
    # Compute flip counts
    flaky_tests = []
    
    for test_name, history in test_history.items():
        flip_count = compute_flip_count(history)
        
        if flip_count > flip_threshold:
            flaky_tests.append({
                "test": test_name,
                "flips": flip_count,
                "history": history,
                "runs": len(history),
            })
    
    # Sort by flip count (descending)
    flaky_tests.sort(key=lambda t: t["flips"], reverse=True)
    
    summary = {
        "total_tests": len(test_history),
        "flaky_tests": len(flaky_tests),
        "window": len(xml_files),
        "flip_threshold": flip_threshold,
    }
    
    return summary, flaky_tests


def main() -> int:
    """Scan for flaky tests.
    
    Returns:
        0 if no flaky tests (or not failing), 1 if flaky tests and FAIL_ON_FLAKY=true
    """
    parser = argparse.ArgumentParser(description="Scan for flaky tests")
    parser.add_argument("--tests-dir", type=pathlib.Path, default="evidence/tests",
                        help="Directory with JUnit XML files")
    parser.add_argument("--output", type=pathlib.Path, default="evidence/regressions/flaky_tests.json",
                        help="Output JSON path")
    parser.add_argument("--window", type=int, default=10,
                        help="Number of runs to analyze")
    parser.add_argument("--flip-threshold", type=int, default=2,
                        help="Number of flips to mark as flaky")
    args = parser.parse_args()
    
    config = get_config()
    
    print()
    print("=" * 100)
    print("FLAKY TEST SCANNER")
    print("=" * 100)
    print()
    print(f"Tests directory:  {args.tests_dir}")
    print(f"Window:           {args.window} runs")
    print(f"Flip threshold:   {args.flip_threshold}")
    print()
    
    if not args.tests_dir.exists():
        print("⚠️  No tests directory found - skipping flaky scan")
        print()
        return 0
    
    # Scan for flaky tests
    print("🔍 Scanning for flaky tests...")
    summary, flaky_tests = scan_flaky_tests(args.tests_dir, args.window, args.flip_threshold)
    print()
    
    print(f"Total tests:  {summary['total_tests']}")
    print(f"Flaky tests:  {summary['flaky_tests']}")
    print(f"Runs analyzed: {summary['window']}")
    print()
    
    if flaky_tests:
        print("🔔 Flaky Tests Detected:")
        print()
        print(f"{'Test':<60s} | {'Flips':>6s} | {'History':<30s}")
        print("-" * 100)
        
        for test in flaky_tests[:10]:  # Show top 10
            test_name = test["test"][:60]
            flips = test["flips"]
            history_str = "".join(["✅" if s == "pass" else "❌" for s in test["history"][-10:]])
            
            print(f"{test_name:<60s} | {flips:>6d} | {history_str:<30s}")
        
        if len(flaky_tests) > 10:
            print(f"... and {len(flaky_tests) - 10} more")
        
        print()
    else:
        print("✅ No flaky tests detected")
        print()
    
    # Write report
    report = {
        "summary": summary,
        "flaky_tests": flaky_tests,
    }
    
    print(f"💾 Writing report to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2)
    print()
    
    # Determine exit code
    if flaky_tests and config["FAIL_ON_FLAKY"]:
        print("❌ Flaky tests detected and FAIL_ON_FLAKY=true")
        print()
        return 1
    else:
        print("✅ Scan complete")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
