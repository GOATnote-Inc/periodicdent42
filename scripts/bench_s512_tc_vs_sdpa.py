#!/usr/bin/env python3
import time, json, numpy as np, torch, torch.nn.functional as F, argparse
from pathlib import Path
from build_v3_release import build_v3_release
from torch.cuda import Stream
import inspect
torch.backends.cuda.matmul.allow_tf32=False

def bench(fn,Q,K,V,s,c,w=5,n=50,streams=False):
  for _ in range(w): fn(Q,K,V,s,c)
  torch.cuda.synchronize()
  ts=[]
  for _ in range(n):
    st = Stream() if streams else None
    if st: 
      with torch.cuda.stream(st): fn(Q,K,V,s,c)
      st.synchronize()
    else:
      fn(Q,K,V,s,c)
    t=time.perf_counter()
    torch.cuda.synchronize()
    ts.append((time.perf_counter()-t)*1e3)
  return float(np.percentile(ts,50)), float(np.percentile(ts,90))

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--streams", action="store_true", help="use per-iteration stream")
  a=ap.parse_args()
  out=Path("cudadent42/artifacts/bench"); out.mkdir(parents=True,exist_ok=True)
  m=build_v3_release()
  # v3: always use forward (Q,K,V,scale,is_causal,config_id)
  v3_fwd = getattr(m,"forward", None)
  assert v3_fwd is not None, "v3 forward not found"
  v3 = lambda Q,K,V,s,c: v3_fwd(Q,K,V,s,c,1)  # config_id=1 (32x64)
  # tc mapping (optional, any available)
  tc = None
  for name in ["forward_tc_64_64_2","forward_tc_128_64_2"]:
    if hasattr(m,name):
      tc_fwd = getattr(m,name)
      tc = lambda Q,K,V,s,c: tc_fwd(Q,K,V,s,c,1)  # config_id=1
      break
  B,H,S,D=2,8,512,64
  torch.manual_seed(42)
  Q=torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)
  K=torch.randn_like(Q); V=torch.randn_like(Q)
  s=1.0/(D**0.5)
  sdpa=lambda q,k,v,sc,ca: F.scaled_dot_product_attention(q,k,v,is_causal=ca,scale=sc)
  res={}
  res["sdpa"]  = dict(zip(("p50_ms","p90_ms"), bench(sdpa,Q,K,V,s,False,streams=a.streams)))
  res["v3"]    = dict(zip(("p50_ms","p90_ms"), bench(v3,Q,K,V,s,False,streams=a.streams)))
  if tc is not None:
    res["tc"]  = dict(zip(("p50_ms","p90_ms"), bench(tc,Q,K,V,s,False,streams=a.streams)))
  (out/"tc_vs_sdpa_s512.json").write_text(json.dumps(res,indent=2))
  print(res)

if __name__=="__main__": main()
