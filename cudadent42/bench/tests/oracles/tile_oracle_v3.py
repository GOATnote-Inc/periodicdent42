#!/usr/bin/env python3
import argparse, torch
torch.backends.cuda.matmul.allow_tf32 = False

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=int,default=0)
    p.add_argument("--noncausal",action="store_true")
    p.add_argument("--causal",action="store_true")
    a=p.parse_args()
    is_causal = a.causal and not a.noncausal

    from build_v3_release import build_v3_release
    m=build_v3_release(debug=True)
    # Map config id to exposed forward; fallback to v3 config 0
    f = getattr(m, "forward_32_64_4_2_1_1", None) or m.forward
    B,H,S,D=2,8,512,64
    torch.manual_seed(42)
    Q=torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)
    K=torch.randn_like(Q); V=torch.randn_like(Q)
    scale=1.0/(D**0.5)
    O=f(Q,K,V,scale,is_causal)
    assert torch.isfinite(O).all()
    print("OK")

if __name__=="__main__": main()
