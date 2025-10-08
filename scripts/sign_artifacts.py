#!/usr/bin/env python3
"""Sign artifacts using Sigstore (cosign) for cryptographic verification.

Uses GitHub OIDC ambient identity (keyless signing) via cosign.
Generates DSSE (Dead Simple Signing Envelope) attestations.

Prerequisites:
    - cosign CLI installed: https://docs.sigstore.dev/cosign/installation
    - Running in GitHub Actions with id-token: write permission

Usage:
    python scripts/sign_artifacts.py --paths evidence/summary/kgi.json evidence/dtp/**/*.json
    python scripts/sign_artifacts.py --paths evidence/ledger/root.txt
"""

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

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


def sign_artifact(
    artifact_path: pathlib.Path,
    output_dir: pathlib.Path,
    keyless: bool = True
) -> Dict[str, Any]:
    """Sign artifact using cosign.
    
    Args:
        artifact_path: Path to artifact to sign
        output_dir: Directory for signature output
        keyless: Use keyless signing (GitHub OIDC)
    
    Returns:
        Dict with signature metadata
    
    Raises:
        subprocess.CalledProcessError: If signing fails
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    
    # Output signature path
    sig_name = f"{artifact_path.name}.intoto.jsonl"
    sig_path = output_dir / sig_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cosign command
    cmd = [
        "cosign",
        "sign-blob",
        "--yes",  # Non-interactive
        str(artifact_path),
        "--output-signature", str(sig_path),
    ]
    
    if keyless:
        # Use ambient OIDC identity (GitHub Actions)
        cmd.extend(["--oidc-issuer", "https://token.actions.githubusercontent.com"])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
        
        return {
            "artifact": str(artifact_path),
            "signature": str(sig_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "cosign keyless (OIDC)",
            "status": "signed",
        }
    
    except subprocess.CalledProcessError as e:
        # Signing failed, but continue with other artifacts
        return {
            "artifact": str(artifact_path),
            "signature": str(sig_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "cosign keyless (OIDC)",
            "status": "failed",
            "error": e.stderr,
        }


def sign_artifacts(
    artifact_paths: List[pathlib.Path],
    output_dir: pathlib.Path,
    keyless: bool = True
) -> Dict[str, Any]:
    """Sign multiple artifacts.
    
    Args:
        artifact_paths: List of artifact paths to sign
        output_dir: Directory for signature outputs
        keyless: Use keyless signing
    
    Returns:
        Dict with signing summary
    """
    results = []
    
    for artifact_path in artifact_paths:
        try:
            result = sign_artifact(artifact_path, output_dir, keyless)
            results.append(result)
        except Exception as e:
            results.append({
                "artifact": str(artifact_path),
                "status": "error",
                "error": str(e),
            })
    
    # Summary
    signed = sum(1 for r in results if r.get("status") == "signed")
    failed = sum(1 for r in results if r.get("status") != "signed")
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_artifacts": len(artifact_paths),
        "signed": signed,
        "failed": failed,
        "results": results,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sign artifacts using Sigstore (cosign)"
    )
    parser.add_argument(
        "--paths",
        type=pathlib.Path,
        nargs="+",
        help="Paths to artifacts to sign"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("evidence/signatures"),
        help="Output directory for signatures (default: evidence/signatures)"
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
    
    if not args.paths:
        parser.print_help()
        return 1
    
    print("\n" + "="*80)
    print("ARTIFACT SIGNING (Sigstore/cosign)".center(80))
    print("="*80 + "\n")
    
    # Check cosign availability
    if not args.skip_cosign_check and not check_cosign_installed():
        print("⚠️  cosign not found - signing disabled")
        print("   Install cosign: https://docs.sigstore.dev/cosign/installation")
        print("   Or run with --skip-cosign-check for testing\n")
        
        # Write placeholder manifest
        placeholder = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "unavailable",
            "reason": "cosign not installed",
            "total_artifacts": len(args.paths),
            "signed": 0,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w") as f:
            json.dump(placeholder, f, indent=2)
        
        print(f"💾 Placeholder manifest written to: {args.manifest}")
        return 0
    
    # Expand glob patterns
    artifact_paths = []
    for pattern in args.paths:
        if "*" in str(pattern):
            # Glob pattern
            parent = pattern.parent
            matches = list(parent.glob(pattern.name))
            artifact_paths.extend(matches)
        else:
            artifact_paths.append(pattern)
    
    # Filter existing files
    existing = [p for p in artifact_paths if p.exists() and p.is_file()]
    missing = [p for p in artifact_paths if not p.exists() or not p.is_file()]
    
    if missing:
        print(f"⚠️  {len(missing)} artifacts not found (skipping):")
        for p in missing[:5]:  # Show first 5
            print(f"   - {p}")
        if len(missing) > 5:
            print(f"   ... and {len(missing) - 5} more")
        print()
    
    if not existing:
        print("❌ No artifacts to sign")
        return 1
    
    print(f"📝 Signing {len(existing)} artifacts...")
    summary = sign_artifacts(existing, args.output_dir, keyless=True)
    
    print(f"\n✅ Signed: {summary['signed']}/{summary['total_artifacts']}")
    if summary['failed'] > 0:
        print(f"⚠️  Failed: {summary['failed']}/{summary['total_artifacts']}")
    
    # Write manifest
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Signature manifest: {args.manifest}")
    print(f"📂 Signatures directory: {args.output_dir}")
    
    print("\n" + "="*80)
    print("✅ Artifact signing complete!")
    print("="*80 + "\n")
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
