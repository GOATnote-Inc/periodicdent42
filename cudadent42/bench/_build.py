#!/usr/bin/env python3
"""
CUDA Kernel Build Helper

Wraps torch.utils.cpp_extension.load() with optimized settings:
- Ninja for parallel builds
- Persistent build cache
- ccache for compilation caching
- L4-optimized flags (SM_89)

Usage:
    from cudadent42.bench._build import build_kernel
    
    module = build_kernel(
        name="fa_s512_bm64",
        sources=["kernels/fa_s512.cu"],
        extra_flags={
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "NUM_WARPS": 4
        }
    )

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-14
"""

import os
import sys
import hashlib
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
import torch.utils.cpp_extension


def _ensure_ninja() -> bool:
    """
    Ensure Ninja is available and in PATH
    
    Returns:
        True if Ninja is available, False otherwise
    """
    # Check if Ninja is installed
    try:
        import ninja
        ninja_available = torch.utils.cpp_extension.is_ninja_available()
    except ImportError:
        ninja_available = False
    
    if not ninja_available:
        print("⚠️  Warning: Ninja not available. Builds will be slower.")
        print("   Install with: pip install ninja")
        print("   Then add to PATH: export PATH=$HOME/.local/bin:$PATH")
    
    return ninja_available


def _config_hash(extra_flags: Dict[str, Any]) -> str:
    """
    Generate hash for build configuration
    
    Args:
        extra_flags: Dictionary of compile-time flags
    
    Returns:
        16-character hash string
    """
    config_str = "_".join(f"{k}{v}" for k, v in sorted(extra_flags.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def build_kernel(
    name: str,
    sources: List[str],
    extra_flags: Optional[Dict[str, Any]] = None,
    build_dir: Optional[str] = None,
    verbose: bool = False,
    use_cache: bool = True
) -> torch.utils.cpp_extension.load:
    """
    Build CUDA kernel with optimized settings
    
    Args:
        name: Module name (unique identifier)
        sources: List of source files (relative or absolute paths)
        extra_flags: Dictionary of compile-time flags (e.g., {"BLOCK_M": 64})
        build_dir: Build directory (default: .torch_build)
        verbose: Print compilation output
        use_cache: Use persistent build cache
    
    Returns:
        Loaded PyTorch extension module
    
    Example:
        >>> module = build_kernel(
        ...     name="my_kernel",
        ...     sources=["kernels/my_kernel.cu"],
        ...     extra_flags={"BLOCK_SIZE": 256}
        ... )
        >>> output = module.forward(input_tensor)
    """
    # Set environment variables for optimized builds
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')  # L4 (Ada, SM_89)
    os.environ.setdefault('MAX_JOBS', str(multiprocessing.cpu_count()))
    os.environ.setdefault('TORCH_CUDA_INCREMENTAL', '0')
    
    # Ensure Ninja is available
    ninja_available = _ensure_ninja()
    
    # Build flags optimized for L4
    nvcc_flags = [
        '-O3',
        '--use_fast_math',
        '-lineinfo',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-Xcompiler=-fno-omit-frame-pointer',
        '-Xcompiler=-fno-common',
        '-gencode=arch=compute_89,code=sm_89',  # L4 only
        '-std=c++17',
        '--threads', str(multiprocessing.cpu_count()),
    ]
    
    # Add extra flags as -D defines
    extra_flags = extra_flags or {}
    for key, val in extra_flags.items():
        nvcc_flags.append(f'-D{key}={val}')
    
    # Determine build directory
    if build_dir is None:
        if use_cache:
            # Persistent cache with config-specific subdirectory
            config_hash = _config_hash(extra_flags)
            build_dir = str(Path.home() / '.torch_cuda_cache' / f"{name}_{config_hash}")
        else:
            # Temporary build directory
            build_dir = f".torch_build/{name}"
    
    # Create build directory
    build_path = Path(build_dir)
    build_path.mkdir(parents=True, exist_ok=True)
    
    # Workaround for PyTorch bug: create lock subdirectory
    (build_path / 'lock').mkdir(exist_ok=True)
    
    # Convert source paths to absolute
    sources_abs = []
    for src in sources:
        src_path = Path(src)
        if not src_path.is_absolute():
            # Try relative to current directory
            if src_path.exists():
                sources_abs.append(str(src_path.resolve()))
            else:
                # Try relative to repo root
                repo_root = Path(__file__).parent.parent.parent
                src_abs = repo_root / src
                if src_abs.exists():
                    sources_abs.append(str(src_abs))
                else:
                    raise FileNotFoundError(f"Source file not found: {src}")
        else:
            sources_abs.append(str(src_path))
    
    if verbose:
        print(f"Building {name}...")
        print(f"  Sources: {sources_abs}")
        print(f"  Build dir: {build_dir}")
        print(f"  Ninja: {ninja_available}")
        print(f"  Cache: {use_cache}")
        print(f"  Extra flags: {extra_flags}")
    
    try:
        # Load or compile extension
        module = torch.utils.cpp_extension.load(
            name=name,
            sources=sources_abs,
            extra_cuda_cflags=nvcc_flags,
            build_directory=str(build_path),
            verbose=verbose,
            with_cuda=True
        )
        
        if verbose:
            print(f"✅ Build succeeded: {name}")
        
        return module
    
    except Exception as e:
        print(f"❌ Build failed: {name}")
        print(f"   Error: {e}")
        print(f"   Sources: {sources_abs}")
        print(f"   Build dir: {build_dir}")
        raise


def main():
    """
    CLI for building kernels
    
    Usage:
        python bench/_build.py --kernel fa_s512 --config "BLOCK_M=64,BLOCK_N=64"
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Build CUDA kernels with optimized settings")
    parser.add_argument("--kernel", required=True, help="Kernel name (e.g., fa_s512)")
    parser.add_argument("--sources", help="Comma-separated source files (default: kernels/{kernel}.cu)")
    parser.add_argument("--config", help="Comma-separated config flags (e.g., BLOCK_M=64,BLOCK_N=64)")
    parser.add_argument("--build-dir", help="Build directory (default: .torch_build)")
    parser.add_argument("--verbose", action="store_true", help="Print compilation output")
    parser.add_argument("--no-cache", action="store_true", help="Disable persistent cache")
    
    args = parser.parse_args()
    
    # Parse sources
    if args.sources:
        sources = args.sources.split(',')
    else:
        sources = [f"cudadent42/bench/kernels/{args.kernel}.cu"]
    
    # Parse config flags
    extra_flags = {}
    if args.config:
        for pair in args.config.split(','):
            if '=' in pair:
                key, val = pair.split('=', 1)
                # Try to parse as int, fallback to string
                try:
                    extra_flags[key.strip()] = int(val.strip())
                except ValueError:
                    extra_flags[key.strip()] = val.strip()
    
    # Build kernel
    print(f"Building kernel: {args.kernel}")
    print(f"  Sources: {sources}")
    print(f"  Config: {extra_flags}")
    print()
    
    try:
        module = build_kernel(
            name=args.kernel,
            sources=sources,
            extra_flags=extra_flags,
            build_dir=args.build_dir,
            verbose=args.verbose,
            use_cache=not args.no_cache
        )
        
        print()
        print(f"✅ Build successful!")
        print(f"   Module: {module.__name__}")
        print(f"   Functions: {[fn for fn in dir(module) if not fn.startswith('_')]}")
        
        return 0
    
    except Exception as e:
        print()
        print(f"❌ Build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

