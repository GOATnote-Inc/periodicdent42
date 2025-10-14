"""
Python bindings for inverted FlashAttention kernel V2 (Tensor Core Version)
TARGET: NVIDIA L4 (Ada Lovelace, SM 8.9)

VERSION: 2.0 (Priority 1 Optimization)
CHANGES: Added wmma::mma_sync for Q@K^T and S@V matmuls
GOAL: 6-8× speedup via FP16 Tensor Cores
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path

# Get kernel paths
KERNEL_DIR = Path(__file__).parent / "kernels"
KERNEL_CU = KERNEL_DIR / "fa_inverted_v2_tensor_cores.cu"
KERNEL_BINDINGS = KERNEL_DIR / "fa_inverted_prod_bindings.cpp"  # Reuse same bindings

# Load the CUDA extension (JIT compilation)
flash_attention_inverted_v2 = torch.utils.cpp_extension.load(
    name="flash_attention_inverted_v2_tc",
    sources=[str(KERNEL_BINDINGS), str(KERNEL_CU)],
    extra_cuda_cflags=[
        "-O3",
        "-use_fast_math",
        "--generate-line-info",
        "-lineinfo",
        "-std=c++17",
        "-arch=sm_89",  # L4 Ada Lovelace
    ],
    verbose=True,
    with_cuda=True,
)


def flash_attention_inverted_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    softmax_scale: float = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Inverted FlashAttention forward pass V2 (Tensor Core Version).
    
    Changes from V1:
    - Uses wmma::mma_sync for Q@K^T matmul (Tensor Cores)
    - Uses wmma::mma_sync for S@V matmul (Tensor Cores)
    - Expected: 6-8× speedup over V1
    
    Args:
        Q: Query tensor [batch_size, seq_len, num_heads, head_dim]
        K: Key tensor [batch_size, seq_len, num_heads, head_dim]
        V: Value tensor [batch_size, seq_len, num_heads, head_dim]
        softmax_scale: Scale factor for attention scores (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor [batch_size, seq_len, num_heads, head_dim]
    """
    
    # Validate inputs
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be on CUDA"
    assert Q.dtype == torch.float16, "Only FP16 supported currently"
    assert Q.shape == K.shape == V.shape, "Q, K, V must have same shape"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), \
        "Inputs must be contiguous"
    
    batch_size, seq_len, num_heads, head_dim = Q.shape
    
    # Validate dimensions
    assert head_dim == 64, "Only HEAD_DIM=64 supported in current version"
    
    # Default softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Call kernel via PyBind11 bindings
    O = flash_attention_inverted_v2.forward(
        Q, K, V, softmax_scale, is_causal
    )
    
    return O


if __name__ == "__main__":
    # Quick smoke test
    print("="*70)
    print(" INVERTED FLASHATTENTION V2 - TENSOR CORE SMOKE TEST")
    print("="*70)
    print("\nRunning smoke test...")
    
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    device = torch.device("cuda:0")
    
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
    V = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
    
    print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    try:
        O = flash_attention_inverted_forward(Q, K, V, is_causal=False)
        print(f"✓ Kernel executed successfully!")
        print(f"  Output shape: {O.shape}")
        print(f"  Output range: [{O.min():.4f}, {O.max():.4f}]")
        print(f"  Output mean: {O.mean():.4f}, std: {O.std():.4f}")
        print("\n✓ SMOKE TEST PASSED (V2 - TENSOR CORE VERSION)")
    except Exception as e:
        print(f"✗ Kernel failed: {e}")
        import traceback
        traceback.print_exc()
        raise
