#!/usr/bin/env python3
import json, sys
from pathlib import Path

p = Path("cudadent42/artifacts/bench/tc_vs_sdpa_s512.json")
if not p.exists():
    print("missing bench json:", p); sys.exit(0)
data = json.loads(p.read_text())
sdpa = data.get("sdpa",{}); v3 = data.get("v3",{}); tc = data.get("tc",{})
def spd(x,y):
    try: return x/y
    except: return None
md = ["# S=512 Bench Summary",
      "",
      "| Impl | p50 (ms) | p90 (ms) | vs SDPA p50 |",
      "|-----:|---------:|---------:|------------:|"]
def row(name,obj):
    if not obj: return
    r = spd(sdpa.get("p50_ms",0), obj.get("p50_ms",0)) if name!="SDPA" else 1.0
    md.append(f"| {name} | {obj.get('p50_ms','-'):.3f} | {obj.get('p90_ms','-'):.3f} | {r if r else '-':.2f}× |")
row("SDPA", sdpa); row("V3", v3); row("TC", tc)
out = Path("cudadent42/artifacts/bench/S512_BENCH_SUMMARY.md")
out.write_text("\n".join(md)+"\n")
print(out)

