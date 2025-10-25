#!/usr/bin/env python3
"""Append-only Merkle ledger for audit trail.

Maintains cryptographic chain of integrity for all artifacts.
Each entry records: timestamp, artifact path, SHA-256 hash, previous root.
After each append, recomputes Merkle root for verification.

Usage:
    python scripts/merkle_ledger.py --append evidence/summary/kgi.json
    python scripts/merkle_ledger.py --verify
    python scripts/merkle_ledger.py --root
"""

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts._config import get_config


def sha256_file(filepath: pathlib.Path) -> str:
    """Compute SHA-256 hash of a file.
    
    Args:
        filepath: Path to file
    
    Returns:
        Hex-encoded SHA-256 hash
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_string(data: str) -> str:
    """Compute SHA-256 hash of a string.
    
    Args:
        data: String to hash
    
    Returns:
        Hex-encoded SHA-256 hash
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_ledger(ledger_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load ledger entries from JSONL file.
    
    Args:
        ledger_path: Path to ledger.jsonl
    
    Returns:
        List of ledger entry dicts
    """
    if not ledger_path.exists():
        return []
    
    entries = []
    with open(ledger_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_merkle_root(entries: List[Dict[str, Any]]) -> str:
    """Compute Merkle root of ledger entries.
    
    Uses a simple hash chain for now (can upgrade to Merkle tree later).
    Root = H(entry_n || H(entry_n-1 || ... || H(entry_1)))
    
    Args:
        entries: List of ledger entry dicts
    
    Returns:
        Hex-encoded Merkle root hash
    """
    if not entries:
        return "0" * 64  # Genesis root (all zeros)
    
    # Hash chain approach
    root = "0" * 64  # Start with genesis
    for entry in entries:
        # Canonical representation: sort keys for determinism
        entry_json = json.dumps(entry, sort_keys=True)
        root = sha256_string(root + entry_json)
    
    return root


def append_entry(
    ledger_path: pathlib.Path,
    artifact_path: pathlib.Path,
    prev_root: Optional[str] = None
) -> Dict[str, Any]:
    """Append new entry to ledger.
    
    Args:
        ledger_path: Path to ledger.jsonl
        artifact_path: Path to artifact being recorded
        prev_root: Previous Merkle root (computed if None)
    
    Returns:
        New ledger entry dict
    """
    # Load existing entries to get previous root
    entries = load_ledger(ledger_path)
    if prev_root is None:
        prev_root = compute_merkle_root(entries)
    
    # Compute artifact hash
    artifact_hash = sha256_file(artifact_path)
    
    # Create new entry
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path),
        "sha256": artifact_hash,
        "prev_root": prev_root,
        "seq": len(entries) + 1,
    }
    
    # Append to ledger (atomic write)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return entry


def verify_ledger(ledger_path: pathlib.Path, verbose: bool = False) -> bool:
    """Verify ledger integrity.
    
    Checks:
    1. Hash chain is valid (prev_root matches computed root)
    2. Artifact files match recorded SHA-256 hashes
    3. Sequence numbers are continuous
    
    Args:
        ledger_path: Path to ledger.jsonl
        verbose: Print detailed verification steps
    
    Returns:
        True if ledger is valid, False otherwise
    """
    entries = load_ledger(ledger_path)
    
    if not entries:
        if verbose:
            print("⚠️  Empty ledger (valid but no entries)")
        return True
    
    if verbose:
        print(f"🔍 Verifying {len(entries)} ledger entries...")
    
    # Check 1: Hash chain
    expected_root = "0" * 64  # Genesis
    for i, entry in enumerate(entries, 1):
        if entry["prev_root"] != expected_root:
            print(f"❌ Entry {i}: Hash chain broken!")
            print(f"   Expected prev_root: {expected_root}")
            print(f"   Actual prev_root:   {entry['prev_root']}")
            return False
        
        # Recompute root after this entry
        entry_json = json.dumps(entry, sort_keys=True)
        expected_root = sha256_string(expected_root + entry_json)
        
        if verbose and i % 10 == 0:
            print(f"   ✅ Verified {i}/{len(entries)} entries")
    
    # Check 2: Sequence numbers
    for i, entry in enumerate(entries, 1):
        if entry.get("seq") != i:
            print(f"❌ Entry {i}: Sequence number mismatch (expected {i}, got {entry.get('seq')})")
            return False
    
    # Check 3: Artifact hashes (optional, only if files still exist)
    artifacts_checked = 0
    artifacts_missing = 0
    for entry in entries:
        artifact_path = pathlib.Path(entry["artifact_path"])
        if artifact_path.exists():
            actual_hash = sha256_file(artifact_path)
            if actual_hash != entry["sha256"]:
                print(f"❌ Artifact tampered: {artifact_path}")
                print(f"   Expected: {entry['sha256']}")
                print(f"   Actual:   {actual_hash}")
                return False
            artifacts_checked += 1
        else:
            artifacts_missing += 1
    
    if verbose:
        print(f"✅ Hash chain valid ({len(entries)} entries)")
        print(f"✅ Sequence numbers continuous")
        print(f"✅ Artifacts verified: {artifacts_checked}")
        if artifacts_missing > 0:
            print(f"⚠️  Artifacts missing: {artifacts_missing} (expected for old entries)")
    
    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Append-only Merkle ledger for audit trail"
    )
    parser.add_argument(
        "--append",
        type=pathlib.Path,
        help="Append artifact to ledger"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ledger integrity"
    )
    parser.add_argument(
        "--root",
        action="store_true",
        help="Compute and display current Merkle root"
    )
    parser.add_argument(
        "--ledger",
        type=pathlib.Path,
        default=pathlib.Path("evidence/ledger/ledger.jsonl"),
        help="Path to ledger file (default: evidence/ledger/ledger.jsonl)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Ensure at least one action
    if not (args.append or args.verify or args.root):
        parser.print_help()
        return 1
    
    print("\n" + "="*80)
    print("MERKLE LEDGER (Append-Only Audit Trail)".center(80))
    print("="*80 + "\n")
    
    # Action: Append
    if args.append:
        if not args.append.exists():
            print(f"❌ Artifact not found: {args.append}")
            return 1
        
        print(f"📝 Appending to ledger: {args.append}")
        entry = append_entry(args.ledger, args.append)
        print(f"✅ Entry {entry['seq']} added")
        print(f"   Artifact: {entry['artifact_path']}")
        print(f"   SHA-256:  {entry['sha256']}")
        print(f"   Prev root: {entry['prev_root'][:16]}...")
        
        # Recompute and save root
        entries = load_ledger(args.ledger)
        root = compute_merkle_root(entries)
        root_path = args.ledger.parent / "root.txt"
        with open(root_path, "w") as f:
            f.write(root)
        print(f"   New root:  {root[:16]}...")
        print(f"💾 Merkle root saved to: {root_path}")
    
    # Action: Verify
    if args.verify:
        print(f"🔍 Verifying ledger: {args.ledger}")
        is_valid = verify_ledger(args.ledger, verbose=args.verbose or True)
        if is_valid:
            entries = load_ledger(args.ledger)
            root = compute_merkle_root(entries)
            print(f"\n✅ Ledger is VALID ({len(entries)} entries)")
            print(f"   Merkle root: {root}")
        else:
            print("\n❌ Ledger verification FAILED")
            return 1
    
    # Action: Root
    if args.root:
        entries = load_ledger(args.ledger)
        root = compute_merkle_root(entries)
        print(f"📊 Current Merkle root ({len(entries)} entries):")
        print(f"   {root}")
        
        # Also save to file
        root_path = args.ledger.parent / "root.txt"
        root_path.parent.mkdir(parents=True, exist_ok=True)
        with open(root_path, "w") as f:
            f.write(root)
        print(f"💾 Saved to: {root_path}")
    
    print("\n" + "="*80)
    print("✅ Ledger operations complete!")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
