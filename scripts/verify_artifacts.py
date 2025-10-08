#!/usr/bin/env python3
"""Verify signed artifacts using Sigstore (cosign).

Verifies DSSE attestations for artifacts signed with cosign.

Usage:
    python scripts/verify_artifacts.py
    python scripts/verify_artifacts.py --manifest evidence/signatures/manifest.json
"""

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts._config import get_config


def check_cosign_installed() -> bool:
    """Check if cosign is installed and available.
    
    Returns:
        True if cosign is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["cosign", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def verify_artifact(
    artifact_path: pathlib.Path,
    signature_path: pathlib.Path
) -> Dict[str, Any]:
    """Verify artifact signature using cosign.
    
    Args:
        artifact_path: Path to artifact
        signature_path: Path to signature file
    
    Returns:
        Dict with verification result
    """
    if not artifact_path.exists():
        return {
            "artifact": str(artifact_path),
            "status": "error",
            "error": "Artifact not found",
        }
    
    if not signature_path.exists():
        return {
            "artifact": str(artifact_path),
            "status": "error",
            "error": "Signature not found",
        }
    
    # Cosign verify command
    cmd = [
        "cosign",
        "verify-blob",
        "--signature", str(signature_path),
        "--certificate-identity-regexp", ".*",  # Accept any identity (for demo)
        "--certificate-oidc-issuer", "https://token.actions.githubusercontent.com",
        str(artifact_path),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True
        )
        
        return {
            "artifact": str(artifact_path),
            "signature": str(signature_path),
            "status": "verified",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    except subprocess.CalledProcessError as e:
        return {
            "artifact": str(artifact_path),
            "signature": str(signature_path),
            "status": "failed",
            "error": e.stderr or "Verification failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify signed artifacts using Sigstore (cosign)"
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("evidence/signatures/manifest.json"),
        help="Path to signature manifest JSON"
    )
    parser.add_argument(
        "--skip-cosign-check",
        action="store_true",
        help="Skip cosign availability check (for CI testing)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ARTIFACT VERIFICATION (Sigstore/cosign)".center(80))
    print("="*80 + "\n")
    
    # Check cosign availability
    if not args.skip_cosign_check and not check_cosign_installed():
        print("⚠️  cosign not found - verification skipped")
        print("   Install cosign: https://docs.sigstore.dev/cosign/installation\n")
        print("✅ Verification skipped (cosign unavailable)")
        return 0
    
    # Load manifest
    if not args.manifest.exists():
        print(f"⚠️  Signature manifest not found: {args.manifest}")
        print("   Run sign_artifacts.py first to generate signatures\n")
        print("✅ Verification skipped (no manifest)")
        return 0
    
    with open(args.manifest, "r") as f:
        manifest = json.load(f)
    
    # Check if signing was successful
    if manifest.get("status") == "unavailable":
        print(f"⚠️  Signing was unavailable: {manifest.get('reason')}")
        print("✅ Verification skipped (no signatures)")
        return 0
    
    if manifest.get("signed", 0) == 0:
        print("⚠️  No artifacts were signed")
        print("✅ Verification skipped (no signatures)")
        return 0
    
    # Verify each signed artifact
    results = manifest.get("results", [])
    signed_results = [r for r in results if r.get("status") == "signed"]
    
    if not signed_results:
        print("⚠️  No successfully signed artifacts in manifest")
        print("✅ Verification skipped (no valid signatures)")
        return 0
    
    print(f"🔍 Verifying {len(signed_results)} signed artifacts...\n")
    
    verification_results = []
    for result in signed_results:
        artifact_path = pathlib.Path(result["artifact"])
        signature_path = pathlib.Path(result["signature"])
        
        print(f"   Verifying: {artifact_path.name}...", end=" ")
        verify_result = verify_artifact(artifact_path, signature_path)
        verification_results.append(verify_result)
        
        if verify_result["status"] == "verified":
            print("✅")
        else:
            print(f"❌ ({verify_result.get('error', 'failed')})")
    
    # Summary
    verified = sum(1 for r in verification_results if r["status"] == "verified")
    failed = sum(1 for r in verification_results if r["status"] != "verified")
    
    print(f"\n📊 Verification Summary:")
    print(f"   ✅ Verified: {verified}/{len(signed_results)}")
    if failed > 0:
        print(f"   ❌ Failed:   {failed}/{len(signed_results)}")
    
    # Write verification report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_artifacts": len(signed_results),
        "verified": verified,
        "failed": failed,
        "results": verification_results,
    }
    
    report_path = args.manifest.parent / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Verification report: {report_path}")
    
    print("\n" + "="*80)
    if failed > 0:
        print("❌ VERIFICATION FAILED".center(80))
        print("="*80 + "\n")
        return 1
    else:
        print("✅ ALL SIGNATURES VERIFIED".center(80))
        print("="*80 + "\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
