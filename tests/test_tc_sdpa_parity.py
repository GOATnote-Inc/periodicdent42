"""
Tensor Core kernel parity tests vs PyTorch SDPA
"""
import torch
import torch.nn.functional as F
import pytest

torch.backends.cuda.matmul.allow_tf32 = False
ATOL, RTOL = 1e-2, 1e-2


@pytest.mark.parametrize("B,H,S,D,causal", [
    (2, 8, 512, 64, False),
    (2, 8, 512, 64, True),
])
def test_tc_s512_parity(B, H, S, D, causal):
    """Test TC kernel (config 64x64) against SDPA reference"""
    try:
        from cudadent42.bench.fa_tc_s512 import flash_attention_tc_s512_forward as tc_fwd
    except (ImportError, AttributeError) as e:
        pytest.skip(f"TC module not available: {e}")
    
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    scale = 1.0 / (D ** 0.5)
    
    # Reference: SDPA (upcast to fp32 for accuracy)
    ref = F.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(),
        is_causal=causal, scale=scale
    ).to(Q.dtype)
    
    # Candidate: TC kernel (config_id=1 → 64x64 tiles)
    out = tc_fwd(Q, K, V, softmax_scale=scale, is_causal=causal, config_id=1)
    
    # Check for numerical issues
    assert not torch.isnan(out).any(), "TC kernel produced NaN"
    assert not torch.isinf(out).any(), "TC kernel produced Inf"
    
    # Check parity
    diff = (out - ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    assert max_diff < ATOL, f"Max diff {max_diff:.6f} exceeds tolerance {ATOL}"
    assert mean_diff < 5 * ATOL, f"Mean diff {mean_diff:.6f} exceeds 5×tolerance {5*ATOL}"
    
    print(f"✅ TC parity (B={B},H={H},S={S},D={D},causal={causal}): "
          f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


@pytest.mark.parametrize("config_id", [1, 2])
def test_tc_both_configs(config_id):
    """Test both TC configs (64x64 and 128x64)"""
    try:
        from cudadent42.bench.fa_tc_s512 import flash_attention_tc_s512_forward as tc_fwd
    except (ImportError, AttributeError) as e:
        pytest.skip(f"TC module not available: {e}")
    
    torch.manual_seed(42)
    B, H, S, D = 2, 8, 512, 64
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K, V = torch.randn_like(Q), torch.randn_like(Q)
    
    out = tc_fwd(Q, K, V, config_id=config_id)
    
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert out.shape == Q.shape
    
    print(f"✅ TC config {config_id} smoke test passed")


if __name__ == "__main__":
    # Quick manual test
    try:
        test_tc_s512_parity(2, 8, 512, 64, False)
        test_tc_s512_parity(2, 8, 512, 64, True)
        test_tc_both_configs(1)
        test_tc_both_configs(2)
        print("\n✅ All TC parity tests passed!")
    except Exception as e:
        print(f"\n❌ TC parity tests failed: {e}")
        import traceback
        traceback.print_exc()

