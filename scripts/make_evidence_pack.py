#!/usr/bin/env python3
"""Generate evidence pack with all provenance artifacts for audit/publication.

Bundles:
- Dataset contracts (checksums, versions)
- Experiment ledger entries
- Coverage reports (JSON/HTML)
- Calibration metrics
- CI run metadata
- Build hashes (double-build verification)
- Documentation (CHANGELOG, PROVENANCE_*)

Output:
- evidence/packs/provenance_pack_{gitsha}_{ts}.{zip|tar.gz}

Usage:
    python scripts/make_evidence_pack.py
    python scripts/make_evidence_pack.py --format tar.gz
    python scripts/make_evidence_pack.py --output custom_pack.zip
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from typing import List, Dict, Any

from _config import get_config


def compute_file_hash(filepath: pathlib.Path) -> str:
    """Compute SHA256 hash of a file.
    
    Args:
        filepath: Path to file
    
    Returns:
        Hex SHA256 hash
    """
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_evidence_files(base_dir: pathlib.Path) -> List[pathlib.Path]:
    """Collect all evidence files for packing.
    
    Args:
        base_dir: Repository root directory
    
    Returns:
        List of Path objects to include in pack
    """
    files = []
    
    # Evidence directory (all JSON/JSONL)
    evidence_dir = base_dir / "evidence"
    if evidence_dir.exists():
        for pattern in ["**/*.json", "**/*.jsonl", "**/*.csv", "**/*.txt"]:
            files.extend(evidence_dir.glob(pattern))
    
    # Coverage reports
    for coverage_file in ["coverage.json", "htmlcov", ".coverage"]:
        path = base_dir / coverage_file
        if path.exists():
            files.append(path)
    
    # Dataset contracts
    if (base_dir / "data_contracts.yaml").exists():
        files.append(base_dir / "data_contracts.yaml")
    
    # Experiment ledger
    ledger_dir = base_dir / "experiments" / "ledger"
    if ledger_dir.exists():
        files.extend(ledger_dir.glob("*.jsonl"))
    
    # Artifacts
    artifact_dir = base_dir / "artifact"
    if artifact_dir.exists():
        for pattern in ["*.json", "*.md", "*.txt", "*.csv"]:
            files.extend(artifact_dir.glob(pattern))
    
    # Documentation
    for doc in ["CHANGELOG_*.md", "PROVENANCE_*.md", "EVIDENCE.md", "README.md"]:
        files.extend(base_dir.glob(doc))
    
    # Build hashes
    for build_file in [".nix-build-hash", "first.hash", "second.hash"]:
        path = base_dir / build_file
        if path.exists():
            files.append(path)
    
    return list(set(files))  # Remove duplicates


def create_manifest(files: List[pathlib.Path], base_dir: pathlib.Path) -> Dict[str, Any]:
    """Create manifest with file listing and checksums.
    
    Args:
        files: List of files included in pack
        base_dir: Repository root for relative paths
    
    Returns:
        Manifest dict
    """
    config = get_config()
    
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": config["GIT_SHA"],
        "git_branch": config["GIT_BRANCH"],
        "ci_run_id": config["CI_RUN_ID"],
        "file_count": len(files),
        "files": []
    }
    
    for file_path in sorted(files):
        try:
            rel_path = file_path.relative_to(base_dir)
            checksum = compute_file_hash(file_path)
            size_bytes = file_path.stat().st_size
            
            manifest["files"].append({
                "path": str(rel_path),
                "sha256": checksum,
                "size_bytes": size_bytes,
            })
        except Exception as e:
            print(f"⚠️  Skipping {file_path}: {e}", file=sys.stderr)
    
    return manifest


def create_zip_pack(files: List[pathlib.Path], output_path: pathlib.Path, 
                    base_dir: pathlib.Path, manifest: Dict[str, Any]) -> None:
    """Create ZIP evidence pack.
    
    Args:
        files: List of files to include
        output_path: Output ZIP path
        base_dir: Repository root for relative paths
        manifest: Manifest dict
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add manifest first
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        
        # Add all files
        for file_path in files:
            try:
                if file_path.is_file():
                    rel_path = file_path.relative_to(base_dir)
                    zf.write(file_path, arcname=rel_path)
                elif file_path.is_dir():
                    # Add directory recursively
                    for item in file_path.rglob("*"):
                        if item.is_file():
                            rel_path = item.relative_to(base_dir)
                            zf.write(item, arcname=rel_path)
            except Exception as e:
                print(f"⚠️  Error adding {file_path}: {e}", file=sys.stderr)


def create_targz_pack(files: List[pathlib.Path], output_path: pathlib.Path,
                      base_dir: pathlib.Path, manifest: Dict[str, Any]) -> None:
    """Create tar.gz evidence pack.
    
    Args:
        files: List of files to include
        output_path: Output tar.gz path
        base_dir: Repository root for relative paths
        manifest: Manifest dict
    """
    with tarfile.open(output_path, "w:gz") as tf:
        # Add manifest first
        import io
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_bytes)
        tf.addfile(info, io.BytesIO(manifest_bytes))
        
        # Add all files
        for file_path in files:
            try:
                if file_path.is_file():
                    rel_path = file_path.relative_to(base_dir)
                    tf.add(file_path, arcname=rel_path)
                elif file_path.is_dir():
                    # Add directory recursively
                    rel_path = file_path.relative_to(base_dir)
                    tf.add(file_path, arcname=rel_path, recursive=True)
            except Exception as e:
                print(f"⚠️  Error adding {file_path}: {e}", file=sys.stderr)


def main() -> int:
    """Generate evidence pack.
    
    Returns:
        0 on success, 1 on error
    """
    parser = argparse.ArgumentParser(description="Generate evidence pack")
    parser.add_argument("--format", choices=["zip", "tar.gz"], default="zip",
                        help="Pack format (default: zip)")
    parser.add_argument("--output", type=pathlib.Path,
                        help="Output path (default: auto-generated)")
    parser.add_argument("--base-dir", type=pathlib.Path, default=pathlib.Path.cwd(),
                        help="Repository root directory")
    args = parser.parse_args()
    
    config = get_config()
    base_dir = args.base_dir.resolve()
    
    print("=" * 80)
    print("EVIDENCE PACK GENERATOR")
    print("=" * 80)
    print()
    print(f"Base directory: {base_dir}")
    print(f"Git SHA:        {config['GIT_SHA']}")
    print(f"Git branch:     {config['GIT_BRANCH']}")
    print(f"CI run ID:      {config['CI_RUN_ID']}")
    print()
    
    # Collect files
    print("📦 Collecting evidence files...")
    files = collect_evidence_files(base_dir)
    print(f"   Found {len(files)} files")
    print()
    
    # Create manifest
    print("📝 Creating manifest...")
    manifest = create_manifest(files, base_dir)
    print(f"   Manifest created with {len(manifest['files'])} entries")
    print()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        git_sha_short = config["GIT_SHA"][:8] if config["GIT_SHA"] != "unknown" else "local"
        filename = f"provenance_pack_{git_sha_short}_{timestamp}.{args.format}"
        
        # Ensure evidence/packs directory exists
        packs_dir = base_dir / "evidence" / "packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        output_path = packs_dir / filename
    
    # Create pack
    print(f"🔨 Creating {args.format} pack...")
    if args.format == "zip":
        create_zip_pack(files, output_path, base_dir, manifest)
    else:  # tar.gz
        create_targz_pack(files, output_path, base_dir, manifest)
    
    # Verify and report
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        checksum = compute_file_hash(output_path)
        
        print()
        print("✅ Evidence pack created successfully!")
        print()
        print(f"   Path:     {output_path}")
        print(f"   Size:     {size_mb:.2f} MB")
        print(f"   SHA256:   {checksum}")
        print(f"   Files:    {len(manifest['files'])}")
        print()
        print("To extract:")
        if args.format == "zip":
            print(f"   unzip {output_path.name}")
        else:
            print(f"   tar -xzf {output_path.name}")
        print()
        
        return 0
    else:
        print("❌ Failed to create evidence pack", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
