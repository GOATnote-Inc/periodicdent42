#!/usr/bin/env python3
"""
Build system for FP8 SDPA Stage-C WMMA kernel with toggles and metadata capture.

Environment variables:
  USE_KV_LUT: 0 (default, direct dequant) or 1 (LUT path)
  DEBUG_PRINT: 0 (default, quiet) or 1 (verbose debug prints)
  TORCH_CUDA_ARCH_LIST: Override default "8.9" for L4
"""

import os
import subprocess
import json
import datetime
from pathlib import Path
from torch.utils.cpp_extension import load

# Toggles from environment
USE_KV_LUT   = int(os.environ.get("USE_KV_LUT", "0"))
DEBUG_PRINT  = int(os.environ.get("DEBUG_PRINT", "0"))
USE_CP_ASYNC = int(os.environ.get("USE_CP_ASYNC", "1"))
USE_WMMA_PV  = int(os.environ.get("USE_WMMA_PV", "1"))  # Default ON (Stage-2 merged)
USE_FUSED_SOFTMAX = int(os.environ.get("USE_FUSED_SOFTMAX", "0"))  # Stage-3: fused softmax in registers (OFF until Step-3 lands)
USE_SMEM_SWIZZLE_XOR = int(os.environ.get("USE_SMEM_SWIZZLE_XOR", "0"))  # Stage-3: XOR swizzle (OFF, Step-2 regressed +6%)
USE_CP_ASYNC_3STAGE = int(os.environ.get("USE_CP_ASYNC_3STAGE", "0"))  # Stage-3: 3-stage pipeline (long seq)
# Stage-5: Warp Specialization + Persistent CTAs
USE_WARP_SPECIALIZATION = int(os.environ.get("USE_WARP_SPECIALIZATION", "0"))  # OFF by default
NUM_PRODUCER_WARPS = int(os.environ.get("NUM_PRODUCER_WARPS", "1"))  # 1 or 2 producers
USE_PERSISTENT_CTA = int(os.environ.get("USE_PERSISTENT_CTA", "0"))  # OFF by default
USE_FAST_EXP = int(os.environ.get("USE_FAST_EXP", "0"))  # OFF by default (breaks correctness)
ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9")

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
KERNEL_DIR = REPO_ROOT / "cudadent42" / "bench" / "kernels"
KERNEL_CU = KERNEL_DIR / "sdpa_fp8_stage_c_wmma.cu"
KERNEL_CPP = KERNEL_DIR / "sdpa_fp8_stage_c_wmma_bindings.cpp"

def build_extension(name="sdpa_fp8_stage_c_wmma", verbose=True):
    """Build CUDA extension with current toggles."""
    
    # Compile flags
    extra_cuda_cflags = [
        "-O3",
        f"-arch=sm_{ARCH_LIST.replace('.', '')}",
        "--use_fast_math",
        "-lineinfo",
        "-Xptxas", "-v",  # Verbose PTXAS for regs/smem
    ]
    
    # Add preprocessor defines
    if USE_KV_LUT:
        extra_cuda_cflags.append("-DUSE_KV_LUT=1")
    if DEBUG_PRINT:
        extra_cuda_cflags.append("-DDEBUG_PRINT=1")
    if USE_CP_ASYNC:
        extra_cuda_cflags.append("-DUSE_CP_ASYNC=1")
    if USE_WMMA_PV:
        extra_cuda_cflags.append("-DUSE_WMMA_PV=1")
    if USE_FUSED_SOFTMAX:
        extra_cuda_cflags.append("-DUSE_FUSED_SOFTMAX=1")
    if USE_SMEM_SWIZZLE_XOR:
        extra_cuda_cflags.append("-DUSE_SMEM_SWIZZLE_XOR=1")
    if USE_CP_ASYNC_3STAGE:
        extra_cuda_cflags.append("-DUSE_CP_ASYNC_3STAGE=1")
    # Stage-5 toggles
    if USE_WARP_SPECIALIZATION:
        extra_cuda_cflags.append("-DUSE_WARP_SPECIALIZATION=1")
        extra_cuda_cflags.append(f"-DNUM_PRODUCER_WARPS={NUM_PRODUCER_WARPS}")
    if USE_PERSISTENT_CTA:
        extra_cuda_cflags.append("-DUSE_PERSISTENT_CTA=1")
    if USE_FAST_EXP:
        extra_cuda_cflags.append("-DUSE_FAST_EXP=1")
    
    print(f"\n{'='*80}")
    print("FP8 SDPA Stage-C WMMA Kernel Build")
    print(f"{'='*80}")
    print(f"  USE_KV_LUT:               {USE_KV_LUT} ({'LUT path' if USE_KV_LUT else 'direct dequant ✓'})")
    print(f"  DEBUG_PRINT:              {DEBUG_PRINT} ({'enabled' if DEBUG_PRINT else 'quiet ✓'})")
    print(f"  USE_CP_ASYNC:             {USE_CP_ASYNC} ({'double-buffer K/V' if USE_CP_ASYNC else 'direct load'})")
    print(f"  USE_WMMA_PV:              {USE_WMMA_PV} ({'WMMA P·V' if USE_WMMA_PV else 'scalar P·V'})")
    print(f"  USE_FUSED_SOFTMAX:        {USE_FUSED_SOFTMAX} ({'fused softmax (no sS)' if USE_FUSED_SOFTMAX else 'Stage-2 baseline'})")
    print(f"  USE_SMEM_SWIZZLE_XOR:     {USE_SMEM_SWIZZLE_XOR} ({'XOR swizzle' if USE_SMEM_SWIZZLE_XOR else 'no swizzle'})")
    print(f"  USE_CP_ASYNC_3STAGE:      {USE_CP_ASYNC_3STAGE} ({'3-stage pipeline' if USE_CP_ASYNC_3STAGE else '2-stage'})")
    print(f"  USE_WARP_SPECIALIZATION:  {USE_WARP_SPECIALIZATION} ({'WS (prod/cons)' if USE_WARP_SPECIALIZATION else 'Stage-2'})")
    if USE_WARP_SPECIALIZATION:
        print(f"    NUM_PRODUCER_WARPS:     {NUM_PRODUCER_WARPS}")
    print(f"  USE_PERSISTENT_CTA:       {USE_PERSISTENT_CTA} ({'persistent' if USE_PERSISTENT_CTA else 'per-tile'})")
    print(f"  USE_FAST_EXP:             {USE_FAST_EXP} ({'fast approx' if USE_FAST_EXP else 'standard __expf ✓'})")
    print(f"  Architecture:             sm_{ARCH_LIST.replace('.', '')}")
    print(f"  Flags:                    {' '.join(extra_cuda_cflags)}")
    print(f"{'='*80}\n")
    
    # Build
    ext = load(
        name=name,
        sources=[str(KERNEL_CU), str(KERNEL_CPP)],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
    
    print(f"\n✅ Extension '{name}' built successfully!\n")
    return ext

def capture_build_metadata(output_dir=None):
    """Capture build metadata for reproducibility."""
    
    # Get git info
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        git_dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD"], cwd=REPO_ROOT
        ) != 0
    except Exception:
        git_sha = "unknown"
        git_branch = "unknown"
        git_dirty = False
    
    # Get device info (requires torch+cuda)
    try:
        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        cuda_version = torch.version.cuda or "N/A"
        sm_version = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    except Exception:
        device_name = "N/A"
        cuda_version = "N/A"
        sm_version = (0, 0)
    
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "build": {
            "USE_KV_LUT": USE_KV_LUT,
            "DEBUG_PRINT": DEBUG_PRINT,
            "USE_CP_ASYNC": USE_CP_ASYNC,
            "USE_WMMA_PV": USE_WMMA_PV,
            "USE_FUSED_SOFTMAX": USE_FUSED_SOFTMAX,
            "USE_SMEM_SWIZZLE_XOR": USE_SMEM_SWIZZLE_XOR,
            "USE_CP_ASYNC_3STAGE": USE_CP_ASYNC_3STAGE,
            "USE_WARP_SPECIALIZATION": USE_WARP_SPECIALIZATION,
            "NUM_PRODUCER_WARPS": NUM_PRODUCER_WARPS,
            "USE_PERSISTENT_CTA": USE_PERSISTENT_CTA,
            "USE_FAST_EXP": USE_FAST_EXP,
            "arch": f"sm_{ARCH_LIST.replace('.', '')}",
            "flags": ["-O3", "--use_fast_math", "-lineinfo"],
        },
        "git": {
            "sha": git_sha,
            "branch": git_branch,
            "dirty": git_dirty,
        },
        "device": {
            "name": device_name,
            "cuda_version": cuda_version,
            "sm_version": f"{sm_version[0]}.{sm_version[1]}",
        },
    }
    
    if output_dir:
        output_path = Path(output_dir) / "build_meta.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Build metadata saved to {output_path}")
    
    return metadata

if __name__ == "__main__":
    # Build extension
    ext = build_extension(verbose=True)
    
    # Capture metadata
    meta = capture_build_metadata()
    
    print("\n" + "="*80)
    print("Build Summary:")
    print("="*80)
    print(f"  Device:   {meta['device']['name']}")
    print(f"  SM:       {meta['device']['sm_version']}")
    print(f"  Git SHA:  {meta['git']['sha'][:8]}")
    print(f"  Branch:   {meta['git']['branch']}")
    print(f"  Dirty:    {meta['git']['dirty']}")
    print("="*80 + "\n")

