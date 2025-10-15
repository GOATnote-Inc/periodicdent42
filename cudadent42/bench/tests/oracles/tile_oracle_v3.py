#!/usr/bin/env python3
import argparse, torch
torch.backends.cuda.matmul.allow_tf32 = False
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=int,default=1)
    p.add_argument("--noncausal",action="store_true"); p.add_argument("--causal",action="store_true")
    a=p.parse_args(); is_causal = a.causal and not a.noncausal
    from build_v3_release import build_v3_release
    m=build_v3_release(debug=True)
    f = getattr(m,"forward", None)
    B,H,S,D=2,8,512,64
    torch.manual_seed(42)
    Q=torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)
    O=f(Q,Q,Q,1.0/(D**0.5),is_causal,1)  # config_id=1 (32x64, STAGES=2)
    assert torch.isfinite(O).all(), "non-finite output"
    print("OK")
if __name__=="__main__": main()
