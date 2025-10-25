#!/usr/bin/env python3
"""GitHub Notification - Post regression reports to PRs.

Features:
- PR comment with regression table
- GitHub Check Run with pass/fail status
- Optional GitHub Issue creation on regression
- Dry-run mode when token absent

Requires: GITHUB_TOKEN, GITHUB_REPOSITORY, PR context
"""

import argparse
import json
import os
import pathlib
import sys


def main() -> int:
    """Send GitHub notifications.
    
    Returns:
        0 on success
    """
    parser = argparse.ArgumentParser(description="Send GitHub notifications")
    parser.add_argument("--regression-report", type=pathlib.Path,
                        default="evidence/regressions/regression_report.json",
                        help="Regression report JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode (print message)")
    args = parser.parse_args()
    
    print()
    print("=" * 100)
    print("GITHUB NOTIFICATIONS")
    print("=" * 100)
    print()
    
    # Check for GitHub context
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPOSITORY")
    
    if not github_token or not github_repo:
        print("⚠️  GITHUB_TOKEN or GITHUB_REPOSITORY not set - dry-run mode")
        print()
        args.dry_run = True
    
    # Load regression report
    if not args.regression_report.exists():
        print(f"⚠️  Regression report not found: {args.regression_report}")
        print()
        return 0
    
    with args.regression_report.open() as f:
        report = json.load(f)
    
    # Generate markdown comment
    comment_lines = []
    comment_lines.append("## 🔍 Regression Detection Report")
    comment_lines.append("")
    comment_lines.append(f"**Status:** {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
    comment_lines.append(f"**Git SHA:** `{report['git_sha']}`")
    comment_lines.append(f"**CI Run:** `{report['ci_run_id']}`")
    comment_lines.append("")
    
    if report["regressions"]:
        comment_lines.append("### ❌ Regressions Detected")
        comment_lines.append("")
        comment_lines.append("| Metric | Baseline | Current | Δ | z | Status |")
        comment_lines.append("|--------|----------|---------|---|---|--------|")
        
        for reg in report["regressions"][:20]:  # First 20
            metric = reg["metric"]
            baseline = reg["baseline_mean"]
            current = reg["current"]
            delta = reg["delta"]
            z = reg["z_score"]
            
            comment_lines.append(f"| {metric} | {baseline:.4f} | {current:.4f} | {delta:+.4f} | {z:+.2f} | ❌ |")
        
        comment_lines.append("")
        comment_lines.append("**Artifacts:**")
        comment_lines.append("- [Regression Report](./evidence/regressions/regression_report.md)")
        comment_lines.append("- [Evidence Pack](./evidence/packs/)")
        comment_lines.append("- [HTML Dashboard](./evidence/report.html)")
        comment_lines.append("")
    
    if report["waivers_applied"]:
        comment_lines.append("### ⚠️ Waivers Applied")
        comment_lines.append("")
        for waiver in report["waivers_applied"]:
            comment_lines.append(f"- **{waiver['metric']}**: {waiver['waiver_reason']}")
            comment_lines.append(f"  - Expires: {waiver['waiver_expires']}")
        comment_lines.append("")
    
    comment_text = "\n".join(comment_lines)
    
    if args.dry_run:
        print("📝 Dry-run mode - would post comment:")
        print()
        print(comment_text)
        print()
        print("✅ Dry-run complete")
        print()
        return 0
    
    # TODO: Implement GitHub API calls
    # - Post PR comment
    # - Create/update Check Run
    # - Optional: create GitHub Issue
    
    print("⚠️  GitHub API integration not yet implemented")
    print("   Showing dry-run output instead:")
    print()
    print(comment_text)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
