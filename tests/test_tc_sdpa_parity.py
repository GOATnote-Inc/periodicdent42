import torch, torch.nn.functional as F, pytest
torch.backends.cuda.matmul.allow_tf32 = False
ATOL,RTOL=1e-2,1e-2
@pytest.mark.parametrize("B,H,S,D,causal",[(2,8,512,64,False),(2,8,512,64,True)])
def test_tc_s512_parity(B,H,S,D,causal):
    from build_v3_release import build_v3_release
    m=build_v3_release()
    f = getattr(m,"forward_tc_64_64_2", None) or getattr(m,"forward_tc_128_64_2", None)
    assert f is not None, "TC forward not available"
    torch.manual_seed(42)
    Q=torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)
    K=torch.randn_like(Q); V=torch.randn_like(Q)
    scale=1.0/(D**0.5)
    ref=F.scaled_dot_product_attention(Q.float(),K.float(),V.float(),is_causal=causal,scale=scale).to(Q.dtype)
    out=f(Q,K,V,scale,causal)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    diff=(out-ref).abs()
    max_diff=diff.max().item()
    assert max_diff<ATOL, f"Max diff {max_diff:.6f} exceeds {ATOL}"
