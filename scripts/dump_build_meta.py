#!/usr/bin/env python3
"""
Capture and display build metadata for reproducibility.

Usage:
    python scripts/dump_build_meta.py
    python scripts/dump_build_meta.py --output results/fp8_wmma_baseline/latest/
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.fp8_sdpa_stage_c_wmma.build import capture_build_metadata

def main():
    parser = argparse.ArgumentParser(description="Capture build metadata")
    parser.add_argument("--output", type=str, help="Output directory for build_meta.json")
    args = parser.parse_args()
    
    # Capture metadata
    meta = capture_build_metadata(output_dir=args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("Build Metadata Summary")
    print("="*80)
    print(f"Timestamp:    {meta['timestamp']}")
    print(f"\nBuild:")
    print(f"  USE_KV_LUT:   {meta['build']['USE_KV_LUT']}")
    print(f"  DEBUG_PRINT:  {meta['build']['DEBUG_PRINT']}")
    print(f"  Architecture: {meta['build']['arch']}")
    print(f"  Flags:        {' '.join(meta['build']['flags'])}")
    print(f"\nGit:")
    print(f"  SHA:          {meta['git']['sha']}")
    print(f"  Branch:       {meta['git']['branch']}")
    print(f"  Dirty:        {meta['git']['dirty']}")
    print(f"\nDevice:")
    print(f"  Name:         {meta['device']['name']}")
    print(f"  CUDA:         {meta['device']['cuda_version']}")
    print(f"  SM:           {meta['device']['sm_version']}")
    print("="*80 + "\n")
    
    if args.output:
        print(f"✅ Metadata saved to {args.output}/build_meta.json\n")

if __name__ == "__main__":
    main()

