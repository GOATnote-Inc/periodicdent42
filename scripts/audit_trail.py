#!/usr/bin/env python3
"""Governance Audit Trail - Track waivers, regressions, and compliance.

Features:
- Combine waivers + regressions + baselines
- Sort by expiry date
- Highlight overdue/unreviewed waivers
- Exit 1 if expired waivers are active

Output: evidence/audit/audit_trail.md
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _config import get_config


def parse_waivers(waiver_file: pathlib.Path) -> List[Dict[str, Any]]:
    """Parse waivers from GOVERNANCE_CHANGE_ACCEPT.yml.
    
    Args:
        waiver_file: Path to waiver YAML file
    
    Returns:
        List of waiver dicts
    """
    waivers = []
    
    if not waiver_file.exists():
        return waivers
    
    try:
        import re
        with waiver_file.open() as f:
            content = f.read()
        
        # Extract waivers section
        waiver_match = re.search(r'waivers:\s*\n((?:  -.*\n(?:    .*\n)*)*)', content)
        if not waiver_match:
            return waivers
        
        waiver_text = waiver_match.group(1)
        
        # Parse each waiver
        waiver_blocks = waiver_text.split('\n  - ')
        for block in waiver_blocks[1:]:  # Skip first empty
            waiver = {}
            for line in block.split('\n'):
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    value = value.strip().strip('"\'')
                    waiver[key] = value
            
            if waiver.get('metric') and waiver.get('expires_at'):
                waivers.append(waiver)
    
    except Exception:
        pass
    
    return waivers


def check_waiver_status(waiver: Dict[str, Any], now: datetime, expire_days: int) -> str:
    """Check waiver status (active, expiring, expired).
    
    Args:
        waiver: Waiver dict
        now: Current datetime
        expire_days: Warning threshold (days)
    
    Returns:
        Status string
    """
    expires_at_str = waiver.get('expires_at', '')
    
    try:
        expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
    except Exception:
        return "⚠️ Invalid expiry"
    
    if now >= expires_at:
        return "❌ Expired"
    
    days_remaining = (expires_at - now).days
    
    if days_remaining <= 7:
        return f"⚠️ Expiring ({days_remaining}d)"
    
    if days_remaining <= expire_days:
        return f"🔔 Warning ({days_remaining}d)"
    
    return f"✅ Active ({days_remaining}d)"


def generate_audit_trail_markdown(
    waivers: List[Dict[str, Any]],
    regressions: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[str, int]:
    """Generate audit trail markdown.
    
    Args:
        waivers: List of waivers
        regressions: List of regressions
        baseline: Baseline dict
        config: Config dict
    
    Returns:
        Tuple of (markdown string, expired_count)
    """
    now = datetime.now(timezone.utc)
    
    lines = []
    lines.append("# Governance Audit Trail")
    lines.append("")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Audit Period:** Last {config['AUDIT_EXPIRE_DAYS']} days")
    lines.append("")
    
    # Summary
    expired_waivers = [w for w in waivers if check_waiver_status(w, now, config['AUDIT_EXPIRE_DAYS']).startswith("❌")]
    expiring_waivers = [w for w in waivers if check_waiver_status(w, now, config['AUDIT_EXPIRE_DAYS']).startswith("⚠️")]
    active_waivers = [w for w in waivers if check_waiver_status(w, now, config['AUDIT_EXPIRE_DAYS']).startswith("✅")]
    
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Waivers:** {len(waivers)}")
    lines.append(f"- **Active:** {len(active_waivers)}")
    lines.append(f"- **Expiring Soon:** {len(expiring_waivers)}")
    lines.append(f"- **Expired:** {len(expired_waivers)} ⚠️")
    lines.append(f"- **Current Regressions:** {len(regressions)}")
    lines.append("")
    
    if expired_waivers:
        lines.append("⚠️ **WARNING:** Expired waivers found! These must be renewed or removed.")
        lines.append("")
    
    # Waiver table
    if waivers:
        lines.append("## Waivers")
        lines.append("")
        lines.append("| Metric | PR | Reason | Expires | Status | Owner | Approver |")
        lines.append("|--------|----|-|--------|--------|-------|----------|")
        
        # Sort by expiry (soonest first)
        sorted_waivers = sorted(waivers, key=lambda w: w.get('expires_at', '9999-12-31'))
        
        for waiver in sorted_waivers:
            metric = waiver.get('metric', 'unknown')
            pr = waiver.get('pr', 'N/A')
            reason = waiver.get('reason', 'N/A')[:50]  # Truncate
            expires = waiver.get('expires_at', 'N/A')
            status = check_waiver_status(waiver, now, config['AUDIT_EXPIRE_DAYS'])
            owner = waiver.get('owner', 'N/A')
            approver = waiver.get('approver', 'N/A')
            
            lines.append(f"| {metric} | {pr} | {reason} | {expires} | {status} | {owner} | {approver} |")
        
        lines.append("")
    else:
        lines.append("## Waivers")
        lines.append("")
        lines.append("*No waivers defined*")
        lines.append("")
    
    # Regressions
    if regressions:
        lines.append("## Active Regressions")
        lines.append("")
        lines.append("| Metric | Current | Baseline | Δ | z-score |")
        lines.append("|--------|---------|----------|---|---------|")
        
        for reg in regressions:
            metric = reg["metric"]
            current = reg["current"]
            baseline_mean = reg["baseline_mean"]
            delta = reg["delta"]
            z = reg["z_score"]
            
            lines.append(f"| {metric} | {current:.4f} | {baseline_mean:.4f} | {delta:+.4f} | {z:+.2f} |")
        
        lines.append("")
    else:
        lines.append("## Active Regressions")
        lines.append("")
        lines.append("*No active regressions*")
        lines.append("")
    
    # Baseline status
    if baseline.get("metrics"):
        lines.append("## Baseline Status")
        lines.append("")
        lines.append(f"- **Updated:** {baseline.get('updated_at', 'unknown')}")
        lines.append(f"- **Window:** {baseline.get('window', 0)} runs")
        lines.append(f"- **Metrics:** {len(baseline.get('metrics', {}))} tracked")
        lines.append("")
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    
    if expired_waivers:
        lines.append(f"1. **URGENT:** Renew or remove {len(expired_waivers)} expired waiver(s)")
        for waiver in expired_waivers:
            lines.append(f"   - {waiver.get('metric', 'unknown')} (PR #{waiver.get('pr', 'N/A')})")
    
    if expiring_waivers:
        lines.append(f"2. **ACTION:** Review {len(expiring_waivers)} waiver(s) expiring soon")
    
    if regressions:
        lines.append(f"3. **REVIEW:** Investigate {len(regressions)} active regression(s)")
    
    if not expired_waivers and not expiring_waivers and not regressions:
        lines.append("✅ No action required. All systems nominal.")
    
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by Periodic Labs Governance Audit System*")
    lines.append("")
    lines.append("**Compliance Notes:**")
    lines.append("- All waivers require code review approval")
    lines.append("- Expired waivers block CI until renewed")
    lines.append(f"- Default expiration: {config['AUDIT_EXPIRE_DAYS']} days")
    lines.append("")
    
    return "\n".join(lines), len(expired_waivers)


def main() -> int:
    """Generate governance audit trail.
    
    Returns:
        0 on success, 1 if expired waivers found
    """
    parser = argparse.ArgumentParser(description="Generate governance audit trail")
    parser.add_argument("--waivers", type=pathlib.Path, default="GOVERNANCE_CHANGE_ACCEPT.yml",
                        help="Waiver file path")
    parser.add_argument("--regression-report", type=pathlib.Path,
                        default="evidence/regressions/regression_report.json",
                        help="Regression report JSON")
    parser.add_argument("--baseline", type=pathlib.Path,
                        default="evidence/baselines/rolling_baseline.json",
                        help="Baseline JSON")
    parser.add_argument("--output", type=pathlib.Path, default="evidence/audit/audit_trail.md",
                        help="Output audit trail markdown")
    args = parser.parse_args()
    
    config = get_config()
    
    print()
    print("=" * 100)
    print("GOVERNANCE AUDIT TRAIL")
    print("=" * 100)
    print()
    
    # Load waivers
    print(f"📂 Loading waivers from: {args.waivers}")
    waivers = parse_waivers(args.waivers)
    print(f"   Found {len(waivers)} waiver(s)")
    
    # Load regressions
    regressions = []
    if args.regression_report.exists():
        with args.regression_report.open() as f:
            report = json.load(f)
            regressions = report.get("regressions", [])
    print(f"   Active regressions: {len(regressions)}")
    
    # Load baseline
    baseline = {}
    if args.baseline.exists():
        with args.baseline.open() as f:
            baseline = json.load(f)
    print()
    
    # Generate audit trail
    print("📝 Generating audit trail...")
    markdown, expired_count = generate_audit_trail_markdown(waivers, regressions, baseline, config)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    
    print(f"💾 Audit trail written to: {args.output}")
    print()
    
    # Print summary
    print("=" * 100)
    print("AUDIT SUMMARY")
    print("=" * 100)
    print()
    print(f"Waivers:           {len(waivers)}")
    print(f"Expired:           {expired_count}")
    print(f"Regressions:       {len(regressions)}")
    print()
    
    if expired_count > 0:
        print("❌ EXPIRED WAIVERS FOUND")
        print()
        print("Action required:")
        print("  1. Renew waivers in GOVERNANCE_CHANGE_ACCEPT.yml")
        print("  2. Or remove expired waivers")
        print("  3. Re-run: make dashboard")
        print()
        return 1
    else:
        print("✅ Audit complete - no expired waivers")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
