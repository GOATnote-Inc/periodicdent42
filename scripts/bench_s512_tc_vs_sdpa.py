#!/usr/bin/env python3
import time, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from build_v3_release import build_v3_release
torch.backends.cuda.matmul.allow_tf32=False

def bench(fn,Q,K,V,s,c,w=20,n=200):
  for _ in range(w):
    fn(Q,K,V,s,c)
    torch.cuda.synchronize()
  ts=[]
  for _ in range(n):
    torch.cuda.synchronize(); t=time.perf_counter()
    fn(Q,K,V,s,c)
    torch.cuda.synchronize()
    ts.append((time.perf_counter()-t)*1e3)
  return float(np.percentile(ts,50)), float(np.percentile(ts,90))

def main():
  out=Path("cudadent42/artifacts/bench"); out.mkdir(parents=True,exist_ok=True)
  m=build_v3_release()
  v3_fn = getattr(m,"forward_32_64_4_2_1_1", None) or getattr(m,"forward", None)
  assert v3_fn is not None, "v3 forward not found"
  v3 = lambda q,k,v,s,c: v3_fn(q,k,v,s,c,1) if getattr(m,"forward", None) == v3_fn else v3_fn(q,k,v,s,c)
  B,H,S,D=2,8,512,64
  torch.manual_seed(42)
  Q=torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)
  K=torch.randn_like(Q); V=torch.randn_like(Q)
  s=1.0/(D**0.5)
  sdpa=lambda q,k,v,sc,ca: F.scaled_dot_product_attention(q,k,v,is_causal=ca,scale=sc)
  res={}
  res["sdpa"]  = dict(zip(("p50_ms","p90_ms"), bench(sdpa,Q,K,V,s,False)))
  res["v3"]    = dict(zip(("p50_ms","p90_ms"), bench(v3,Q,K,V,s,False)))
  (out/"tc_vs_sdpa_s512.json").write_text(json.dumps(res,indent=2))
  print(res)

if __name__=="__main__": main()
