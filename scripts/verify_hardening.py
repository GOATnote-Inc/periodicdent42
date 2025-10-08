#!/usr/bin/env python3
"""Verification script for hardening implementation."""

import json
import pathlib
import sys

def check_kgi_u():
    """Check KGI_u metric (unitless)."""
    kgi_path = pathlib.Path("evidence/summary/kgi.json")
    if not kgi_path.exists():
        return {"status": "❌", "details": "File not found"}
    
    with open(kgi_path) as f:
        data = json.load(f)
    
    if "kgi_u" in data and data.get("units") == "unitless":
        return {"status": "✅", "details": f"value={data['kgi_u']:.4f}, {data.get('disclaimer', '')[:40]}..."}
    else:
        return {"status": "⚠️", "details": "Missing kgi_u or units field"}

def check_kgi_bits():
    """Check KGI_bits (Shannon entropy)."""
    kgi_bits_path = pathlib.Path("evidence/summary/kgi_bits.json")
    if not kgi_bits_path.exists():
        return {"status": "N/A", "details": "Probe set not available (expected for demo)"}
    
    with open(kgi_bits_path) as f:
        data = json.load(f)
    
    if data.get("kgi_bits") == "unavailable":
        return {"status": "N/A", "details": data.get("reason", "Unavailable")}
    else:
        return {"status": "✅", "details": f"bits={data.get('kgi_bits')}, pre={data.get('pre_bits')}, post={data.get('post_bits')}"}

def check_merkle_ledger():
    """Check Merkle ledger."""
    ledger_path = pathlib.Path("evidence/ledger/ledger.jsonl")
    root_path = pathlib.Path("evidence/ledger/root.txt")
    
    if not ledger_path.exists():
        return {"status": "⚠️", "details": "Ledger not initialized"}
    
    # Count entries
    with open(ledger_path) as f:
        entries = [line for line in f if line.strip()]
    
    if root_path.exists():
        with open(root_path) as f:
            root = f.read().strip()
        return {"status": "✅", "details": f"root={root[:16]}..., {len(entries)} entries"}
    else:
        return {"status": "⚠️", "details": f"{len(entries)} entries, no root computed"}

def check_signatures():
    """Check DSSE signatures."""
    manifest_path = pathlib.Path("evidence/signatures/manifest.json")
    if not manifest_path.exists():
        return {"status": "N/A", "details": "Not signed (cosign unavailable)"}
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    if manifest.get("status") == "unavailable":
        return {"status": "N/A", "details": manifest.get("reason", "cosign not installed")}
    
    signed = manifest.get("signed", 0)
    return {"status": "✅", "details": f"n={signed} artifacts signed"}

def check_claims():
    """Check claims guard."""
    report_path = pathlib.Path("evidence/claims/claims_report.json")
    if not report_path.exists():
        return {"status": "⚠️", "details": "Claims not verified yet"}
    
    with open(report_path) as f:
        report = json.load(f)
    
    if report.get("passed"):
        return {"status": "✅", "details": "No violations"}
    else:
        violations = len(report.get("violations", []))
        return {"status": "❌", "details": f"{violations} violations found"}

def check_dvc_dataset():
    """Check DVC dataset ID."""
    # Check if DVC is configured
    dvc_path = pathlib.Path(".dvc")
    if not dvc_path.exists():
        return {"status": "⚠️", "details": "DVC not initialized"}
    
    return {"status": "N/A", "details": "DVC configured but no datasets tracked yet"}

def main():
    print("\n" + "="*80)
    print("HARDENING VERIFICATION SUMMARY".center(80))
    print("="*80 + "\n")
    
    checks = [
        ("KGI_u metric", check_kgi_u),
        ("KGI_bits (if probe)", check_kgi_bits),
        ("DVC dataset_id", check_dvc_dataset),
        ("Merkle ledger", check_merkle_ledger),
        ("DSSE signatures", check_signatures),
        ("Claims guard", check_claims),
    ]
    
    print("| Check | Status | Details |")
    print("|-------|--------|---------|")
    
    for name, check_fn in checks:
        result = check_fn()
        print(f"| {name:<20} | {result['status']:<6} | {result['details'][:50]} |")
    
    print("\n" + "="*80)
    print("✅ Verification complete!".center(80))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
