#!/usr/bin/env python3
"""
FlashCore Build System

Compiles CUDA kernels with PyTorch C++ extensions.

Environment Variables:
    CUDA_ARCH: Target CUDA architecture (default: "8.9" for L4)
    DEBUG: Enable debug symbols and verbose output (default: 0)
    VERBOSE: Verbose build output (default: 0)

Usage:
    python build.py                    # Build baseline kernel
    DEBUG=1 python build.py            # Build with debug symbols
    CUDA_ARCH=7.5 python build.py      # Build for Tesla T4 (sm_75)

Returns:
    Compiled PyTorch extension module (flashcore_baseline)
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from torch.utils.cpp_extension import load
    import torch
except ImportError:
    print("ERROR: PyTorch not found. Please install PyTorch with CUDA support:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# CUDA architecture (default: 8.9 for L4)
CUDA_ARCH = os.environ.get("CUDA_ARCH", "8.9")

# Debug mode
DEBUG = int(os.environ.get("DEBUG", "0"))

# Verbose build output
VERBOSE = int(os.environ.get("VERBOSE", "0"))

# Paths
REPO_ROOT = Path(__file__).parent
KERNEL_DIR = REPO_ROOT / "kernels"
KERNEL_CU = KERNEL_DIR / "flashcore_baseline.cu"
KERNEL_CPP = KERNEL_DIR / "bindings.cpp"

# Validate files exist
if not KERNEL_CU.exists():
    print(f"ERROR: Kernel file not found: {KERNEL_CU}")
    sys.exit(1)
if not KERNEL_CPP.exists():
    print(f"ERROR: Bindings file not found: {KERNEL_CPP}")
    sys.exit(1)

# ============================================================================
# Build Function
# ============================================================================

def build_baseline(verbose=None):
    """Build FlashCore baseline kernel.
    
    Args:
        verbose (bool): Override VERBOSE env var
    
    Returns:
        PyTorch extension module with .forward() method
    """
    
    if verbose is None:
        verbose = bool(VERBOSE)
    
    # CUDA compile flags
    extra_cuda_cflags = [
        "-O3" if not DEBUG else "-O0",
        f"-arch=sm_{CUDA_ARCH.replace('.', '')}",
        "--use_fast_math",
        "-lineinfo",
        "-Xptxas", "-v",  # Verbose PTXAS: show registers, shared memory
    ]
    
    if DEBUG:
        extra_cuda_cflags.extend([
            "-G",           # Device debug
            "-DDEBUG=1",    # Enable debug prints in kernel
        ])
    
    # C++ compile flags
    extra_cflags = ["-O3" if not DEBUG else "-O0 -g"]
    
    # Print build configuration
    print(f"\n{'='*80}")
    print("FlashCore Baseline Kernel Build")
    print(f"{'='*80}")
    print(f"  CUDA Arch:      sm_{CUDA_ARCH.replace('.', '')}")
    print(f"  Debug Mode:     {'Yes' if DEBUG else 'No'}")
    print(f"  Verbose:        {'Yes' if verbose else 'No'}")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CUDA Version:   {torch.version.cuda}")
    print(f"  Kernel:         {KERNEL_CU.name}")
    print(f"  Bindings:       {KERNEL_CPP.name}")
    print(f"{'='*80}\n")
    
    # Build with PyTorch JIT
    try:
        ext = load(
            name="flashcore_baseline",
            sources=[str(KERNEL_CU), str(KERNEL_CPP)],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=extra_cflags,
            verbose=verbose,
        )
    except Exception as e:
        print(f"\n{'='*80}")
        print("❌ Build Failed")
        print(f"{'='*80}")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check CUDA toolkit is installed: nvcc --version")
        print("  2. Check PyTorch has CUDA support: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  3. Try verbose mode: VERBOSE=1 python build.py")
        print("  4. Check CUDA architecture matches GPU: nvidia-smi")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("✅ Build Complete")
    print(f"{'='*80}")
    print(f"Module: flashcore_baseline")
    print(f"Forward: flashcore_baseline.forward(Q, K, V, scale)")
    print(f"{'='*80}\n")
    
    return ext

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available in PyTorch")
        print("Build will proceed but kernel won't be usable")
        print("\nInstall CUDA-enabled PyTorch:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Build kernel
    ext = build_baseline()
    
    # Quick sanity check (if CUDA available)
    if torch.cuda.is_available():
        print("Running quick sanity check...")
        try:
            B, H, S, D = 1, 1, 32, 64
            Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
            K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
            V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
            scale = 1.0 / (D ** 0.5)
            
            O = ext.forward(Q, K, V, scale)
            
            assert O.shape == (B, H, S, D), f"Output shape mismatch: {O.shape}"
            assert not torch.isnan(O).any(), "Output contains NaN"
            assert not torch.isinf(O).any(), "Output contains Inf"
            
            print("✅ Sanity check passed!")
        except Exception as e:
            print(f"⚠️  Sanity check failed: {e}")
            print("Kernel compiled but may have runtime issues")
    
    print("\nNext steps:")
    print("  python -m pytest tests/test_correctness.py -v")
    print("  python benchmarks/benchmark_latency.py --shape mission")

