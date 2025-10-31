"""JIT compilation utilities for BlackwellSparseK kernels."""

import os
import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from .config import Config, get_default_config


def get_build_info() -> Dict[str, Any]:
    """Get build environment information."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_arch"] = f"sm_{major}{minor}"
    
    return info


def detect_cuda_arch() -> str:
    """Auto-detect CUDA compute capability."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    major, minor = torch.cuda.get_device_capability(0)
    
    # Map to supported architectures
    if major == 9 and minor == 0:
        return "90a"  # H100
    elif major == 10 and minor == 0:
        return "100"  # Blackwell B200
    else:
        raise RuntimeError(
            f"Unsupported GPU architecture: sm_{major}{minor}. "
            "BlackwellSparseK requires sm_90a (H100) or sm_100 (Blackwell)."
        )


def build_kernel(
    config: Optional[Config] = None,
    force_rebuild: bool = False,
    verbose: bool = False
) -> Any:
    """
    Build BlackwellSparseK CUDA extension using JIT compilation.
    
    Args:
        config: Build configuration (uses default if None)
        force_rebuild: Force recompilation even if cached
        verbose: Print compilation output
    
    Returns:
        Compiled PyTorch extension module
    
    Example:
        >>> from blackwell_sparsek.core import build_kernel
        >>> kernel = build_kernel(verbose=True)
        >>> output = kernel.attention_forward(Q, K, V, scale=0.125)
    """
    if config is None:
        config = get_default_config()
    
    # Auto-detect architecture if enabled
    if config.auto_detect_arch:
        try:
            detected_arch = detect_cuda_arch()
            if verbose:
                print(f"Auto-detected CUDA architecture: sm_{detected_arch}")
            config.cuda_arch = detected_arch
        except RuntimeError as e:
            if verbose:
                print(f"Auto-detection failed: {e}")
                print(f"Using configured architecture: sm_{config.cuda_arch}")
    
    # Source files
    package_dir = Path(__file__).parent.parent
    kernel_dir = package_dir / "kernels"
    
    sources = [
        str(kernel_dir / "attention_fmha.cu"),
        str(kernel_dir / "kernel_dispatch.cu"),
        str(kernel_dir / "kernel_bindings.cpp"),
    ]
    
    # Verify sources exist
    for src in sources:
        if not Path(src).exists():
            raise FileNotFoundError(f"Kernel source not found: {src}")
    
    # CUDA compilation flags
    cuda_flags = [
        "-O3" if not config.debug_mode else "-O0",
        "--use_fast_math" if config.use_fast_math else "",
        "-lineinfo" if config.enable_profiling else "",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-std=c++17",
        f"-gencode=arch=compute_{config.cuda_arch.replace('a', '')},code=sm_{config.cuda_arch}",
        f"-I{config.cutlass_path}/include",
        f"-DBLOCK_M={config.block_m}",
        f"-DBLOCK_N={config.block_n}",
        f"-DNUM_STAGES={config.num_stages}",
        f"-DNUM_WARPS={config.num_warps}",
    ]
    
    if config.debug_mode:
        cuda_flags.extend(["-G", "-DDEBUG=1"])
    
    # C++ compilation flags
    cxx_flags = [
        "-O3" if not config.debug_mode else "-O0 -g",
        "-fPIC",
    ]
    
    # Filter empty flags
    cuda_flags = [f for f in cuda_flags if f]
    
    if verbose:
        print("=" * 80)
        print("BlackwellSparseK Kernel Build")
        print("=" * 80)
        print(f"  CUDA Arch:      sm_{config.cuda_arch}")
        print(f"  CUTLASS Path:   {config.cutlass_path}")
        print(f"  Build Dir:      {config.build_dir}")
        print(f"  Debug Mode:     {config.debug_mode}")
        print(f"  Fast Math:      {config.use_fast_math}")
        print(f"  Profiling:      {config.enable_profiling}")
        print("=" * 80)
    
    # Build with torch.utils.cpp_extension
    from torch.utils.cpp_extension import load
    
    module = load(
        name="blackwell_sparsek_kernels",
        sources=sources,
        extra_cuda_cflags=cuda_flags,
        extra_cflags=cxx_flags,
        build_directory=config.build_dir,
        verbose=verbose,
        with_cuda=True,
    )
    
    if verbose:
        print("✅ Build successful!")
    
    return module

