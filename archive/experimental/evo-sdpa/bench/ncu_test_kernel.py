#!/usr/bin/env python3
"""
NCU Test Script - Minimal kernel launcher for profiling
This script is called by sudo ncu, so it must handle its own imports
"""
import sys
import os

# Add venv to path if not already there
venv_path = os.path.expanduser("~/venv/lib/python3.11/site-packages")
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# Now import torch
try:
    import torch
except ImportError:
    print("ERROR: torch not found. Ensure venv is activated.", file=sys.stderr)
    sys.exit(1)

# Import our benchmark harness
sys.path.insert(0, os.path.dirname(__file__))
from bench_sdpa import build_ext, run_case

if __name__ == "__main__":
    print("[Building kernel...]", file=sys.stderr)
    mod = build_ext()
    
    print("[Running kernel (1,8,512,64)...]", file=sys.stderr)
    run_case(mod, B=1, H=8, L=512, d=64, causal=False, verbose=False, iters=1)
    
    print("[Complete]", file=sys.stderr)

