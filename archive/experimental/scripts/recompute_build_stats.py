#!/usr/bin/env python3
"""Recompute build statistics from available evidence.

Usage:
    python scripts/recompute_build_stats.py --output reports/build_stats.csv
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Recompute build statistics")
    parser.add_argument("--output", default="reports/build_stats.csv",
                        help="Output CSV file")
    args = parser.parse_args()
    
    print("=" * 80)
    print("BUILD STATISTICS RECOMPUTATION")
    print("=" * 80)
    print()
    
    # Check if Nix is available
    try:
        result = subprocess.run(["nix", "--version"], capture_output=True, text=True)
        nix_version = result.stdout.strip()
        print(f"✓ Nix installed: {nix_version}")
    except FileNotFoundError:
        print("✗ Nix not found (cannot recompute builds)")
        nix_version = "Not installed"
    
    # Check flake.nix
    flake_path = Path("flake.nix")
    if flake_path.exists():
        print(f"✓ flake.nix found ({flake_path.stat().st_size} bytes)")
        flake_lines = len(flake_path.read_text().splitlines())
        print(f"  Lines: {flake_lines}")
    else:
        print("✗ flake.nix not found")
        flake_lines = 0
    
    # Check CI workflow
    ci_nix_path = Path(".github/workflows/ci-nix.yml")
    if ci_nix_path.exists():
        print(f"✓ ci-nix.yml found ({ci_nix_path.stat().st_size} bytes)")
        ci_lines = len(ci_nix_path.read_text().splitlines())
        print(f"  Lines: {ci_lines}")
    else:
        print("✗ ci-nix.yml not found")
        ci_lines = 0
    
    # Statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "nix_version": nix_version,
        "flake_lines": flake_lines,
        "ci_lines": ci_lines,
        "flake_exists": flake_path.exists(),
        "ci_exists": ci_nix_path.exists(),
    }
    
    # Attempt to get flake metadata
    if flake_path.exists() and "Not installed" not in nix_version:
        try:
            result = subprocess.run(
                ["nix", "flake", "metadata", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                stats["flake_locked"] = metadata.get("locked", {}).get("locked", False)
                print(f"✓ Flake metadata retrieved")
            else:
                stats["flake_locked"] = None
                print(f"✗ Flake metadata failed: {result.stderr[:100]}")
        except Exception as e:
            stats["flake_locked"] = None
            print(f"✗ Flake metadata error: {e}")
    
    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)
    
    print()
    print(f"✓ Statistics written to {output_path}")
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
