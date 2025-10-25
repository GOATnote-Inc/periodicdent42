#!/usr/bin/env python3
"""Result Regression Detection for Scientific Computing

Automatically validates that numerical results haven't regressed beyond
acceptable tolerances. Compares current results against DVC-tracked baselines.

Phase 3 Week 9 Day 5-7: Result Regression Detection

Usage:
    python scripts/check_regression.py --current validation_branin.json \\
                                       --baseline data/baselines/branin_baseline.json \\
                                       --tolerance 1e-10

Author: GOATnote Autonomous Research Lab Initiative
Date: October 6, 2025
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RegressionResult:
    """Result of regression check for a single field."""
    field: str
    current_value: float
    baseline_value: float
    difference: float
    relative_diff: float
    passed: bool
    tolerance: float


@dataclass
class RegressionReport:
    """Complete regression check report."""
    passed: bool
    total_checks: int
    failed_checks: int
    results: List[RegressionResult]
    tolerance: float
    current_file: str
    baseline_file: str


class RegressionChecker:
    """Check for numerical regressions in scientific results."""
    
    def __init__(self, tolerance: float = 1e-10):
        """Initialize regression checker.
        
        Args:
            tolerance: Maximum acceptable absolute difference
        """
        self.tolerance = tolerance
    
    def load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file with results.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    def extract_numerical_fields(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        """Extract all numerical fields from nested dictionary.
        
        Args:
            data: Dictionary with results
            prefix: Prefix for nested keys
            
        Returns:
            Flat dictionary of field paths to numerical values
        """
        fields = {}
        
        for key, value in data.items():
            field_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (int, float)):
                fields[field_path] = float(value)
            elif isinstance(value, dict):
                # Recursively extract from nested dicts
                nested = self.extract_numerical_fields(value, field_path)
                fields.update(nested)
            elif isinstance(value, list):
                # Extract from list items if numerical
                for i, item in enumerate(value):
                    if isinstance(item, (int, float)):
                        fields[f"{field_path}[{i}]"] = float(item)
                    elif isinstance(item, dict):
                        nested = self.extract_numerical_fields(item, f"{field_path}[{i}]")
                        fields.update(nested)
        
        return fields
    
    def compare_values(
        self,
        field: str,
        current: float,
        baseline: float,
        tolerance: float
    ) -> RegressionResult:
        """Compare current value against baseline.
        
        Args:
            field: Field name
            current: Current value
            baseline: Baseline value
            tolerance: Maximum acceptable difference
            
        Returns:
            RegressionResult with comparison details
        """
        difference = abs(current - baseline)
        
        # Compute relative difference (handle division by zero)
        if baseline != 0:
            relative_diff = difference / abs(baseline)
        else:
            relative_diff = difference if current != 0 else 0.0
        
        passed = difference <= tolerance
        
        return RegressionResult(
            field=field,
            current_value=current,
            baseline_value=baseline,
            difference=difference,
            relative_diff=relative_diff,
            passed=passed,
            tolerance=tolerance
        )
    
    def check_regression(
        self,
        current_path: Path,
        baseline_path: Path,
        tolerance: Optional[float] = None
    ) -> RegressionReport:
        """Check for regressions between current and baseline results.
        
        Args:
            current_path: Path to current results JSON
            baseline_path: Path to baseline results JSON
            tolerance: Override default tolerance
            
        Returns:
            RegressionReport with all comparisons
        """
        tol = tolerance if tolerance is not None else self.tolerance
        
        # Load both files
        current_data = self.load_json(current_path)
        baseline_data = self.load_json(baseline_path)
        
        # Extract numerical fields
        current_fields = self.extract_numerical_fields(current_data)
        baseline_fields = self.extract_numerical_fields(baseline_data)
        
        # Find common fields
        common_fields = set(current_fields.keys()) & set(baseline_fields.keys())
        
        # Check each field
        results = []
        for field in sorted(common_fields):
            result = self.compare_values(
                field,
                current_fields[field],
                baseline_fields[field],
                tol
            )
            results.append(result)
        
        # Compute summary
        failed = [r for r in results if not r.passed]
        passed = len(failed) == 0
        
        return RegressionReport(
            passed=passed,
            total_checks=len(results),
            failed_checks=len(failed),
            results=results,
            tolerance=tol,
            current_file=str(current_path),
            baseline_file=str(baseline_path)
        )
    
    def format_report(self, report: RegressionReport) -> str:
        """Format regression report as human-readable string.
        
        Args:
            report: RegressionReport to format
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("RESULT REGRESSION DETECTION REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Current:  {report.current_file}")
        lines.append(f"Baseline: {report.baseline_file}")
        lines.append(f"Tolerance: {report.tolerance:.2e}")
        lines.append("")
        lines.append(f"Status: {'✅ PASSED' if report.passed else '❌ FAILED'}")
        lines.append(f"Checks: {report.total_checks} total, {report.failed_checks} failed")
        lines.append("")
        
        if report.failed_checks > 0:
            lines.append("FAILED CHECKS:")
            lines.append("-" * 80)
            for result in report.results:
                if not result.passed:
                    lines.append(f"  Field: {result.field}")
                    lines.append(f"    Current:  {result.current_value:.15e}")
                    lines.append(f"    Baseline: {result.baseline_value:.15e}")
                    lines.append(f"    Diff:     {result.difference:.15e} (>{result.tolerance:.2e})")
                    lines.append(f"    Rel Diff: {result.relative_diff:.2%}")
                    lines.append("")
        else:
            lines.append("All checks passed! ✅")
            lines.append("")
            lines.append("Sample comparisons:")
            for result in report.results[:5]:  # Show first 5
                lines.append(f"  {result.field}: {result.difference:.2e} (✓)")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_json(self, report: RegressionReport, output_path: Path):
        """Export regression report as JSON.
        
        Args:
            report: RegressionReport to export
            output_path: Path to output JSON file
        """
        data = {
            "passed": report.passed,
            "total_checks": report.total_checks,
            "failed_checks": report.failed_checks,
            "tolerance": report.tolerance,
            "current_file": report.current_file,
            "baseline_file": report.baseline_file,
            "results": [
                {
                    "field": r.field,
                    "current_value": r.current_value,
                    "baseline_value": r.baseline_value,
                    "difference": r.difference,
                    "relative_diff": r.relative_diff,
                    "passed": r.passed
                }
                for r in report.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_html(self, report: RegressionReport, output_path: Path):
        """Export regression report as HTML visualization.
        
        Args:
            report: RegressionReport to export
            output_path: Path to output HTML file
        """
        failed_results = [r for r in report.results if not r.passed]
        passed_results = [r for r in report.results if r.passed]
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Regression Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: {'#d32f2f' if not report.passed else '#388e3c'}; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .passed {{ color: #388e3c; }}
        .failed {{ color: #d32f2f; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .fail-row {{ background: #ffebee; }}
    </style>
</head>
<body>
    <h1>Regression Detection Report</h1>
    
    <div class="summary">
        <p><strong>Status:</strong> <span class="{'failed' if not report.passed else 'passed'}">
            {'❌ FAILED' if not report.passed else '✅ PASSED'}
        </span></p>
        <p><strong>Current:</strong> {report.current_file}</p>
        <p><strong>Baseline:</strong> {report.baseline_file}</p>
        <p><strong>Tolerance:</strong> {report.tolerance:.2e}</p>
        <p><strong>Total Checks:</strong> {report.total_checks}</p>
        <p><strong>Failed Checks:</strong> {report.failed_checks}</p>
    </div>
    
    {"<h2>Failed Checks</h2>" if failed_results else "<h2>All Checks Passed!</h2>"}
    
    {self._generate_table(failed_results) if failed_results else ""}
    
    <h2>Sample Passed Checks</h2>
    {self._generate_table(passed_results[:10])}
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _generate_table(self, results: List[RegressionResult]) -> str:
        """Generate HTML table for results."""
        if not results:
            return "<p>No results to display.</p>"
        
        rows = []
        for r in results:
            row_class = 'fail-row' if not r.passed else ''
            rows.append(f"""
            <tr class="{row_class}">
                <td>{r.field}</td>
                <td>{r.current_value:.6e}</td>
                <td>{r.baseline_value:.6e}</td>
                <td>{r.difference:.6e}</td>
                <td>{r.relative_diff:.2%}</td>
                <td>{'❌' if not r.passed else '✅'}</td>
            </tr>
            """)
        
        return f"""
        <table>
            <tr>
                <th>Field</th>
                <th>Current</th>
                <th>Baseline</th>
                <th>Difference</th>
                <th>Relative Diff</th>
                <th>Status</th>
            </tr>
            {''.join(rows)}
        </table>
        """


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Check for numerical regressions in scientific results"
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current results JSON"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline results JSON"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Maximum acceptable absolute difference (default: 1e-10)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Export report as JSON"
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Export report as HTML"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero code if regression detected"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not args.current.exists():
        print(f"ERROR: Current file not found: {args.current}", file=sys.stderr)
        sys.exit(1)
    
    if not args.baseline.exists():
        print(f"ERROR: Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    
    # Run regression check
    checker = RegressionChecker(tolerance=args.tolerance)
    report = checker.check_regression(args.current, args.baseline)
    
    # Print report
    print(checker.format_report(report))
    
    # Export if requested
    if args.output_json:
        checker.export_json(report, args.output_json)
        print(f"\n✅ JSON report exported to: {args.output_json}")
    
    if args.output_html:
        checker.export_html(report, args.output_html)
        print(f"✅ HTML report exported to: {args.output_html}")
    
    # Exit with appropriate code
    if args.fail_on_regression and not report.passed:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
