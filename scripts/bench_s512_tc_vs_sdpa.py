#!/usr/bin/env python3
import time, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from build_v3_release import build_v3_release
import inspect
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

def adapt_sig(fn):
  # Try calling with 5 args, fallback to 6 (Q,K,V,scale,is_causal,config_id=1)
  def wrapper(Q,K,V,s,c):
    try:
      return fn(Q,K,V,s,c)
    except TypeError as e:
      if "incompatible function arguments" in str(e) or "takes 6 positional arguments" in str(e):
        return fn(Q,K,V,s,c,1)
      raise
  return wrapper

def main():
  out=Path("cudadent42/artifacts/bench"); out.mkdir(parents=True,exist_ok=True)
  m=build_v3_release()
  # v3 mapping (prefer 32x64 release sym; fallback to generic forward)
  v3_raw = getattr(m,"forward_32_64_4_2_1_1", None) or getattr(m,"forward", None)
  assert v3_raw is not None, "v3 forward not found"
  v3 = adapt_sig(v3_raw)
  # tc mapping (optional, any available)
  tc_raw = None
  for name in ["forward_tc_64_64_2","forward_tc_128_64_2"]:
    if hasattr(m,name): tc_raw = getattr(m,name); break
  tc = adapt_sig(tc_raw) if tc_raw else None
  B,H,S,D=2,8,512,64
  torch.manual_seed(42)
  Q=torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)
  K=torch.randn_like(Q); V=torch.randn_like(Q)
  s=1.0/(D**0.5)
  sdpa=lambda q,k,v,sc,ca: F.scaled_dot_product_attention(q,k,v,is_causal=ca,scale=sc)
  res={}
  res["sdpa"]  = dict(zip(("p50_ms","p90_ms"), bench(sdpa,Q,K,V,s,False)))
  res["v3"]    = dict(zip(("p50_ms","p90_ms"), bench(v3,Q,K,V,s,False)))
  if tc is not None:
    res["tc"]  = dict(zip(("p50_ms","p90_ms"), bench(tc,Q,K,V,s,False)))
  (out/"tc_vs_sdpa_s512.json").write_text(json.dumps(res,indent=2))
  print(res)

if __name__=="__main__": main()
