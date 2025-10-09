#!/usr/bin/env python3
"""
Generate provenance manifest with SHA-256 checksums.

Creates a manifest of all data files, model checkpoints, and configuration files
with cryptographic hashes for reproducibility verification.

Usage:
    python scripts/generate_provenance.py \
        --output evidence/phase10/tier2_clean/MANIFEST.sha256
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
import sys

def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def generate_manifest(
    repo_root: Path,
    output_path: Path
):
    """Generate provenance manifest with SHA-256 checksums"""
    
    print("="*70)
    print("PROVENANCE MANIFEST GENERATION")
    print("="*70)
    
    manifest = {
        'generated': datetime.now().isoformat(),
        'repository': str(repo_root),
        'files': {}
    }
    
    # Categories of files to track
    categories = {
        'data': ['data/**/*.csv', 'data/**/*.json'],
        'configs': ['configs/**/*.yaml', 'configs/**/*.json', 'pyproject.toml'],
        'checkpoints': ['checkpoints/**/*.pkl', 'checkpoints/**/*.pt'],
        'results': ['evidence/**/*.json', 'evidence/**/*results*.json'],
        'scripts': ['scripts/**/*.py']
    }
    
    total_files = 0
    
    for category, patterns in categories.items():
        print(f"\n📁 {category.upper()}")
        manifest['files'][category] = []
        
        for pattern in patterns:
            for filepath in repo_root.glob(pattern):
                if filepath.is_file():
                    try:
                        checksum = compute_sha256(filepath)
                        relative_path = filepath.relative_to(repo_root)
                        size_bytes = filepath.stat().st_size
                        
                        file_info = {
                            'path': str(relative_path),
                            'sha256': checksum,
                            'size_bytes': size_bytes,
                            'modified': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                        }
                        
                        manifest['files'][category].append(file_info)
                        total_files += 1
                        
                        print(f"  ✅ {relative_path}")
                        print(f"     SHA-256: {checksum[:16]}...")
                    except Exception as e:
                        print(f"  ❌ Error processing {filepath}: {e}")
    
    # Save manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Also save as text file
    text_output = output_path.with_suffix('.txt')
    with open(text_output, 'w') as f:
        f.write(f"# Provenance Manifest\n")
        f.write(f"# Generated: {manifest['generated']}\n")
        f.write(f"# Repository: {manifest['repository']}\n\n")
        
        for category, files in manifest['files'].items():
            f.write(f"\n## {category.upper()} ({len(files)} files)\n\n")
            for file_info in files:
                f.write(f"{file_info['sha256']}  {file_info['path']}\n")
    
    print("\n" + "="*70)
    print("MANIFEST COMPLETE")
    print("="*70)
    print(f"Total files: {total_files}")
    print(f"JSON: {output_path}")
    print(f"Text: {text_output}")
    
    return manifest

def verify_manifest(manifest_path: Path) -> bool:
    """Verify all files in manifest match their checksums"""
    
    print("\n" + "="*70)
    print("VERIFYING MANIFEST")
    print("="*70)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    repo_root = Path(manifest['repository'])
    all_valid = True
    total_checked = 0
    total_mismatches = 0
    total_missing = 0
    
    for category, files in manifest['files'].items():
        print(f"\n📁 {category.upper()}")
        
        for file_info in files:
            filepath = repo_root / file_info['path']
            expected_hash = file_info['sha256']
            
            if not filepath.exists():
                print(f"  ❌ MISSING: {file_info['path']}")
                all_valid = False
                total_missing += 1
                continue
            
            actual_hash = compute_sha256(filepath)
            total_checked += 1
            
            if actual_hash == expected_hash:
                print(f"  ✅ MATCH: {file_info['path']}")
            else:
                print(f"  ❌ MISMATCH: {file_info['path']}")
                print(f"     Expected: {expected_hash[:16]}...")
                print(f"     Actual:   {actual_hash[:16]}...")
                all_valid = False
                total_mismatches += 1
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"Checked: {total_checked}")
    print(f"Matches: {total_checked - total_mismatches}")
    print(f"Mismatches: {total_mismatches}")
    print(f"Missing: {total_missing}")
    
    if all_valid:
        print("\n✅ ALL FILES VERIFIED")
        return True
    else:
        print("\n❌ VERIFICATION FAILED")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate provenance manifest')
    parser.add_argument('--output', type=Path, 
                       default=Path('evidence/phase10/tier2_clean/MANIFEST.sha256'))
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing manifest instead of generating')
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    
    if args.verify:
        if not args.output.exists():
            print(f"❌ Manifest not found: {args.output}")
            sys.exit(1)
        
        valid = verify_manifest(args.output)
        sys.exit(0 if valid else 1)
    else:
        manifest = generate_manifest(repo_root, args.output)
        
        # Automatically verify after generation
        print("\n🔍 Verifying generated manifest...")
        verify_manifest(args.output)

if __name__ == '__main__':
    main()

