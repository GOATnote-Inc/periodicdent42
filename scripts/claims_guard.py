#!/usr/bin/env python3
"""CI guardrails to block unverified performance claims.

Scans documentation for risky claims and ensures evidence exists.

Blocked claims without proof:
- "10x acceleration/faster" → requires evidence/studies/ab_speed.json
- "bits/run" or "Shannon entropy" → requires evidence/summary/kgi_bits.json
- "1-2 runs early warning" → requires evidence/studies/regression_validation.json

Usage:
    python scripts/claims_guard.py
    python scripts/claims_guard.py --docs docs/ --strict
"""

import argparse
import json
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts._config import get_config


# Claim patterns and required evidence
CLAIM_RULES = [
    {
        "id": "10x_acceleration",
        "patterns": [
            r"10x\s+(acceleration|faster|speedup)",
            r"10×\s+(acceleration|faster|speedup)",
            r"10\s*[xX]\s+(acceleration|faster|speedup)",
        ],
        "evidence_path": "evidence/studies/ab_speed.json",
        "description": "10x acceleration claim",
        "required_fields": ["methodology", "control_group", "treatment_group", "speedup_factor"],
    },
    {
        "id": "bits_per_run",
        "patterns": [
            r"bits?\s+per\s+run",
            r"bits?/run",
            r"Shannon\s+entropy\s+reduction",
            r"uncertainty\s+reduced.*bits",
        ],
        "evidence_path": "evidence/summary/kgi_bits.json",
        "description": "Bits/run claim (Shannon entropy)",
        "required_fields": ["kgi_bits", "pre_bits", "post_bits"],
        "allowed_if_unavailable": True,  # OK if kgi_bits says "unavailable"
    },
    {
        "id": "early_warning",
        "patterns": [
            r"1-2\s+runs?\s+early\s+warning",
            r"1–2\s+runs?\s+early\s+warning",
            r"early\s+warning.*1-2\s+runs?",
        ],
        "evidence_path": "evidence/studies/regression_validation.json",
        "description": "1-2 runs early warning claim",
        "required_fields": ["methodology", "precision", "recall", "lead_time_runs"],
    },
]


def scan_file_for_claims(
    filepath: pathlib.Path,
    claim_rules: List[Dict[str, Any]]
) -> List[Tuple[str, str, int]]:
    """Scan file for claims matching patterns.
    
    Args:
        filepath: Path to file to scan
        claim_rules: List of claim rule dicts
    
    Returns:
        List of (claim_id, matched_text, line_number) tuples
    """
    if not filepath.exists():
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (UnicodeDecodeError, PermissionError):
        return []
    
    findings = []
    for line_num, line in enumerate(lines, 1):
        for rule in claim_rules:
            for pattern in rule["patterns"]:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    findings.append((rule["id"], match.group(0), line_num))
    
    return findings


def check_evidence_exists(rule: Dict[str, Any], base_dir: pathlib.Path) -> Dict[str, Any]:
    """Check if evidence exists for a claim.
    
    Args:
        rule: Claim rule dict
        base_dir: Repository root directory
    
    Returns:
        Dict with validation result
    """
    evidence_path = base_dir / rule["evidence_path"]
    
    if not evidence_path.exists():
        return {
            "valid": False,
            "reason": f"Evidence file not found: {rule['evidence_path']}",
        }
    
    # Load and validate evidence
    try:
        with open(evidence_path, "r") as f:
            evidence = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "valid": False,
            "reason": f"Invalid JSON in evidence file: {e}",
        }
    
    # Check if unavailable is allowed
    if rule.get("allowed_if_unavailable") and evidence.get(rule["required_fields"][0]) == "unavailable":
        return {
            "valid": True,
            "reason": "Claim allowed (evidence unavailable, but declared)",
            "evidence": evidence,
        }
    
    # Check required fields
    missing_fields = [f for f in rule.get("required_fields", []) if f not in evidence]
    if missing_fields:
        return {
            "valid": False,
            "reason": f"Evidence missing required fields: {', '.join(missing_fields)}",
        }
    
    # Additional validation for specific claims
    if rule["id"] == "10x_acceleration":
        speedup = evidence.get("speedup_factor", 0)
        if speedup < 10.0:
            return {
                "valid": False,
                "reason": f"Speedup factor {speedup:.1f}x is less than claimed 10x",
            }
    
    if rule["id"] == "early_warning":
        lead_time = evidence.get("lead_time_runs", 0)
        if lead_time < 1 or lead_time > 2:
            return {
                "valid": False,
                "reason": f"Lead time {lead_time} runs does not match claim of 1-2 runs",
            }
    
    return {
        "valid": True,
        "reason": "Evidence validated",
        "evidence": evidence,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CI guardrails to block unverified claims"
    )
    parser.add_argument(
        "--docs",
        type=pathlib.Path,
        nargs="+",
        default=[pathlib.Path("docs"), pathlib.Path("README.md"), pathlib.Path("DISCOVERY_KERNEL_COMPLETE.md")],
        help="Paths to documentation to scan"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode (fail on any unverified claim)"
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=pathlib.Path("evidence/claims/claims_report.json"),
        help="Path to claims report JSON"
    )
    
    args = parser.parse_args()
    base_dir = pathlib.Path.cwd()
    
    print("\n" + "="*80)
    print("CLAIMS GUARDRAILS (Verify Performance Claims)".center(80))
    print("="*80 + "\n")
    
    # Collect all files to scan
    files_to_scan = []
    for doc_path in args.docs:
        if doc_path.is_file():
            files_to_scan.append(doc_path)
        elif doc_path.is_dir():
            # Scan HTML and markdown files
            files_to_scan.extend(doc_path.glob("*.html"))
            files_to_scan.extend(doc_path.glob("*.md"))
    
    print(f"📂 Scanning {len(files_to_scan)} documentation files...")
    print()
    
    # Scan for claims
    all_findings = {}
    for filepath in files_to_scan:
        findings = scan_file_for_claims(filepath, CLAIM_RULES)
        if findings:
            all_findings[str(filepath)] = findings
    
    if not all_findings:
        print("✅ No performance claims found in documentation")
        print("\n" + "="*80)
        print("✅ CLAIMS VALIDATION PASSED".center(80))
        print("="*80 + "\n")
        return 0
    
    # Group findings by claim ID
    claims_by_id = {}
    for filepath, findings in all_findings.items():
        for claim_id, matched_text, line_num in findings:
            if claim_id not in claims_by_id:
                claims_by_id[claim_id] = []
            claims_by_id[claim_id].append({
                "file": filepath,
                "line": line_num,
                "text": matched_text,
            })
    
    # Validate each claim
    violations = []
    for claim_id, occurrences in claims_by_id.items():
        rule = next(r for r in CLAIM_RULES if r["id"] == claim_id)
        
        print(f"🔍 Checking: {rule['description']}")
        print(f"   Found {len(occurrences)} occurrence(s)")
        
        validation = check_evidence_exists(rule, base_dir)
        
        if validation["valid"]:
            print(f"   ✅ {validation['reason']}")
        else:
            print(f"   ❌ {validation['reason']}")
            violations.append({
                "claim_id": claim_id,
                "description": rule["description"],
                "occurrences": occurrences,
                "validation": validation,
            })
        print()
    
    # Generate report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_scanned": len(files_to_scan),
        "claims_found": sum(len(v) for v in claims_by_id.values()),
        "unique_claims": len(claims_by_id),
        "violations": violations,
        "passed": len(violations) == 0,
    }
    
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"💾 Claims report: {args.report}")
    
    # Summary
    print("\n" + "="*80)
    if violations:
        print("❌ CLAIMS VALIDATION FAILED".center(80))
        print("="*80 + "\n")
        print(f"Found {len(violations)} unverified claim(s):\n")
        for v in violations:
            print(f"  • {v['description']}")
            print(f"    Reason: {v['validation']['reason']}")
            print(f"    Locations:")
            for occ in v['occurrences'][:3]:  # Show first 3
                print(f"      - {occ['file']}:{occ['line']}")
            if len(v['occurrences']) > 3:
                print(f"      ... and {len(v['occurrences']) - 3} more")
            print()
        
        if args.strict:
            return 1
        else:
            print("⚠️  Continuing (use --strict to fail on violations)")
            return 0
    else:
        print("✅ ALL CLAIMS VERIFIED".center(80))
        print("="*80 + "\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
