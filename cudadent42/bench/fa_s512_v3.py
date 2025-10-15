"""
FlashAttention S=512 V3 - Memory-Optimized Kernel
TARGET: ≤0.255 ms (≥20% faster than V2's 0.3184 ms)

Key Features:
- cp.async 2-stage pipeline for K/V
- Register-only softmax (no SMEM S)
- Persistent blocks for L2 reuse
- half2 vectorized loads/stores
- Swizzled SMEM for bank conflict elimination
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path
import os

# Get kernel paths
KERNEL_DIR = Path(__file__).parent / "kernels"
KERNEL_CU = KERNEL_DIR / "fa_s512_v3.cu"
KERNEL_BINDINGS = KERNEL_DIR / "fa_s512_v3_bindings.cpp"

# JIT compile extension (cached after first build)
_extension = None

def _load_extension():
    global _extension
    if _extension is not None:
        return _extension
    
    print("Compiling V3 kernel (this may take 1-2 minutes on first run)...")
    
    _extension = torch.utils.cpp_extension.load(
        name="flash_attention_s512_v3",
        sources=[str(KERNEL_BINDINGS), str(KERNEL_CU)],
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "--generate-line-info",
            "-std=c++17",
            "-arch=sm_89",  # L4 Ada Lovelace
            "-I" + str(KERNEL_DIR),  # For detail/ includes
        ],
        extra_cflags=["-O3"],
        verbose=False,
        with_cuda=True,
    )
    
    print("✅ V3 kernel compiled successfully!")
    return _extension


def flash_attention_s512_v3_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    softmax_scale: float = None,
    is_causal: bool = False,
    config_id: int = 1,  # Which template instantiation (1, 2, or 3)
) -> torch.Tensor:
    """
    FlashAttention S=512 V3 forward pass (memory-optimized).
    
    Args:
        Q: Query [B, H, S, D] (FP16, S=512, D=64)
        K: Key [B, H, S, D]
        V: Value [B, H, S, D]
        softmax_scale: Scale for attention scores (default: 1/sqrt(D))
        is_causal: Whether to apply causal masking
        config_id: Which config to use (1=32x64, 2=32x32, 3=48x64)
        
    Returns:
        Output [B, H, S, D]
    """
    # Validate
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be on CUDA"
    assert Q.dtype == torch.float16, "Only FP16 supported"
    assert Q.shape == K.shape == V.shape, "Q, K, V must have same shape"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    
    B, H, S, D = Q.shape
    assert S == 512, "This kernel is specialized for S=512"
    assert D == 64, "Only HEAD_DIM=64 supported"
    assert config_id in [1, 2, 3, 4, 5], f"config_id must be 1-5 (got {config_id})"
    
    # Default softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)
    
    # Load extension
    ext = _load_extension()
    
    # Call kernel
    O = ext.forward(Q, K, V, softmax_scale, is_causal, config_id)
    
    return O


# Config metadata for search/reporting
CONFIGS = {
    1: {"BLOCK_M": 32, "BLOCK_N": 64, "WARPS": 6, "STAGES": 2, "SWIZZLE": True, "HALF2": True},
    2: {"BLOCK_M": 32, "BLOCK_N": 32, "WARPS": 6, "STAGES": 2, "SWIZZLE": True, "HALF2": True},
    3: {"BLOCK_M": 48, "BLOCK_N": 64, "WARPS": 8, "STAGES": 2, "SWIZZLE": True, "HALF2": True},
}


if __name__ == "__main__":
    # Smoke test
    print("="*70)
    print(" FlashAttention S=512 V3 - Memory-Optimized Smoke Test")
    print("="*70)
    
    B, H, S, D = 2, 8, 512, 64
    device = torch.device("cuda:0")
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    K = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    V = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    
    print(f"\nInput: B={B}, H={H}, S={S}, D={D}")
    
    # Test all 3 configs
    for config_id in [1, 2, 3]:
        cfg = CONFIGS[config_id]
        print(f"\nTesting Config {config_id}: BLOCK_M={cfg['BLOCK_M']}, BLOCK_N={cfg['BLOCK_N']}, WARPS={cfg['WARPS']}")
        
        try:
            O = flash_attention_s512_v3_forward(Q, K, V, config_id=config_id)
            print(f"  ✅ Config {config_id} executed successfully!")
            print(f"     Output shape: {O.shape}")
            print(f"     Output range: [{O.min():.4f}, {O.max():.4f}]")
            
            # Quick correctness check vs SDPA
            Q_sdpa = Q.permute(0, 2, 1, 3)  # B,S,H,D
            K_sdpa = K.permute(0, 2, 1, 3)
            V_sdpa = V.permute(0, 2, 1, 3)
            O_sdpa = torch.nn.functional.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa)
            O_sdpa = O_sdpa.permute(0, 2, 1, 3)  # B,H,S,D
            
            max_diff = (O - O_sdpa).abs().max().item()
            print(f"     Max diff vs SDPA: {max_diff:.6f}")
            
            if max_diff < 0.01:
                print(f"     ✅ Correctness looks good!")
            else:
                print(f"     ⚠️  Large error: {max_diff}")
        except Exception as e:
            print(f"  ❌ Config {config_id} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Smoke test complete!")
