"""
FlashAttention Tensor Core kernel for S=512, D=64 (CUTLASS-backed)

PROTOTYPE STATUS:
- Working foundation with online softmax
- Expect 2-5× slower than SDPA initially (needs refinement)
- Specialized for S=512, D=64 only
"""

import torch
import os
from pathlib import Path
from torch.utils.cpp_extension import load

def build_fa_tc_s512(debug=False):
    """
    Build FlashAttention Tensor Core kernel using PyTorch's JIT compilation.
    
    Args:
        debug: If True, compile with -G -lineinfo for debugging
    
    Returns:
        Compiled module with forward() function
    """
    # Find source files
    kernel_dir = Path(__file__).parent / "kernels"
    cutlass_dir = Path(__file__).parent.parent.parent / "third_party" / "cutlass"
    
    sources = [
        str(kernel_dir / "fa_tc_s512.cu"),
        str(kernel_dir / "fa_tc_s512_bindings.cpp"),
    ]
    
    # Compile flags
    extra_cuda_cflags = [
        "-O3",
        "-use_fast_math",
        "-lineinfo",
        "-Xptxas", "-v",
        "--expt-relaxed-constexpr",
        f"-I{cutlass_dir}/include",
        f"-I{cutlass_dir}/tools/util/include",
    ]
    
    if debug:
        extra_cuda_cflags.extend(["-G", "-DDEBUG_TC"])
    
    # Use Ninja for faster compilation
    os.environ.setdefault("USE_NINJA", "1")
    
    print("Compiling TC kernel (this may take 2-3 minutes on first run with CUTLASS headers)...")
    print(f"CUTLASS path: {cutlass_dir}")
    
    module = load(
        name="flash_attention_tc_s512",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-std=c++17"],
        verbose=True,
    )
    
    print("✅ TC kernel compiled successfully!")
    return module

def flash_attention_tc_s512_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    softmax_scale: float = None,
    is_causal: bool = False,
    config_id: int = 1,
) -> torch.Tensor:
    """
    FlashAttention forward pass using Tensor Cores.
    
    Args:
        Q, K, V: (B, H, S, D) tensors, fp16, CUDA, contiguous
        softmax_scale: Scale for QK^T (default: 1/sqrt(D))
        is_causal: Apply causal mask
        config_id: 1=64x64, 2=128x64
    
    Returns:
        O: (B, H, S, D) output tensor
        
    Constraints:
        - S must be 512 (specialized kernel)
        - D must be 64
        - dtype must be float16
    """
    # Validation
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be on CUDA"
    assert Q.dtype == torch.float16, "Only FP16 supported"
    assert Q.shape == K.shape == V.shape, "Q, K, V must have same shape"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    
    B, H, S, D = Q.shape
    assert S == 512, f"This kernel is specialized for S=512 (got {S})"
    assert D == 64, f"Only HEAD_DIM=64 supported (got {D})"
    assert config_id in [1, 2], f"config_id must be 1 or 2 (got {config_id})"
    
    # Default softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)
    
    # Build module (cached after first call)
    if not hasattr(flash_attention_tc_s512_forward, '_module'):
        flash_attention_tc_s512_forward._module = build_fa_tc_s512()
    
    module = flash_attention_tc_s512_forward._module
    
    # Forward pass
    return module.forward(Q, K, V, softmax_scale, is_causal, config_id)


__all__ = ['flash_attention_tc_s512_forward', 'build_fa_tc_s512']

