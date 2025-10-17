import itertools, json, subprocess, os, statistics
from pathlib import Path
from time import perf_counter
import torch
from bench.layernorm.build_ln import build_ln

def bench_once(params):
    env=os.environ.copy()
    for k,v in params.items(): env[k]=str(v)
    code="import bench.layernorm.bench_ln as b"  # prints max_diff and us
    t0=perf_counter()
    out=subprocess.check_output(["python","-c",code],env=env,text=True)
    t1=perf_counter()
    lines=[s for s in out.splitlines() if s.strip()]
    md=float(lines[0].split('=')[-1])
    us=float(lines[1])
    return md,us,t1-t0

grid=dict(
    THREADS=[128,256,512],
    ROWS_PER_CTA=[1,2],
    VEC_WIDTH=[2,4],
    USE_WARP=[1],
)
keys=list(grid.keys())
best=None; logs=[]
for vals in itertools.product(*grid.values()):
    p=dict(zip(keys,vals))
    md,us,_=bench_once(p)
    rec=dict(params=p,max_diff=md,time_us=us)
    logs.append(rec)
    if md<2e-3 and (best is None or us<best["time_us"]): best=rec
Path("evidence").mkdir(exist_ok=True)
Path("evidence/ln_sweep.json").write_text(json.dumps({"best":best,"logs":logs},indent=2))
print(best)

