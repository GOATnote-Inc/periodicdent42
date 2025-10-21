#!/usr/bin/env python3
"""
Summarize bench + NCU into reports/summary.md
- Table: baselines vs top‑3 candidates (latency, speedup×, mem GB (approx), peak GB/s (from NCU), occupancy, TC util)
- NCU highlights & bottleneck analysis (textual)
- Autotune search space and convergence (text)
- Risks + next actions
"""
import json, glob, os, statistics, math, shutil
from pathlib import Path

def load_jsons(pat):
    res = []
    for p in glob.glob(pat):
        try:
            res.append(json.load(open(p)))
        except Exception:
            pass
    return res

def fmt(x, digits=2):
    return ("%.2f" % x) if isinstance(x,(int,float)) and math.isfinite(x) else "-"

def main():
    Path("reports").mkdir(exist_ok=True)
    bench = load_jsons("artifacts/bench/*.json")
    ncu = json.load(open("artifacts/ncu/summary.json")) if os.path.exists("artifacts/ncu/summary.json") else {"entries":[]}
    tune = json.load(open("artifacts/tune/topk.json")) if os.path.exists("artifacts/tune/topk.json") else {"topk":[]}

    # Build a quick lookup for NCU metrics by name
    ncu_map = {e["name"]: e["metrics"] for e in ncu.get("entries",[])}

    lines = []
    lines += ["# SDPA WS — Evaluation Summary", ""]
    # Environment
    if os.path.exists("artifacts/ENV.json"):
        env = json.load(open("artifacts/ENV.json"))
        lines += ["**Environment**",
                  f"- Python: {env.get('python')}  |  PyTorch: {env.get('torch')}  |  CUDA: {env.get('cuda')}",
                  f"- Device: {env.get('device')}  |  SM: {env.get('sm')}",
                  ""]
    # Table header
    lines += ["## Baselines vs Top‑3 Candidates (mission shape)",
              "",
              "| Variant | p50 (μs) | Speedup vs A | Speedup vs B | TC util % | Occupancy % | DRAM %peak | Notes |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]

    # Extract mission-shape entries
    def mission_rows(tag):
        return [r for r in bench if r["shape"]["name"]=="mission" and r["backend"]==tag]

    rows = []
    for tag,label in [("baseline_a","Baseline A"),("baseline_b","Baseline B"),
                      ("candidate_triton_ws","Cand: Triton WS"),("candidate_triton_flashlike","Cand: Triton FlashLike"),
                      ("candidate_cuda_stub","Cand: CUDA Stub")]:
        rs = mission_rows(tag)
        if not rs:
            continue
        # choose latest (by ts)
        r = sorted(rs, key=lambda x: x["ts"], reverse=True)[0]
        name = label
        p50 = r["latency"]["p50_us"]
        sA  = r["speedup"]["vs_A"]
        sB  = r["speedup"]["vs_B"]
        # NCU mapping (use matching names where possible)
        if tag=="baseline_a":
            m = ncu_map.get("baseline", {})
        elif tag.startswith("candidate_"):
            idx = {"candidate_triton_ws":1,"candidate_triton_flashlike":2,"candidate_cuda_stub":3}.get(tag, None)
            m = ncu_map.get(f"candidate_{idx}", {})
        else:
            m = {}
        tc = m.get("tc_util_pct","-")
        occ = m.get("achieved_occupancy_pct","-")
        dram = m.get("dram_bw_pct","-")
        rows.append((name,p50,sA,sB,tc,occ,dram,""))

    for row in rows:
        lines.append("| " + " | ".join([str(row[0]), *(fmt(x) for x in row[1:7]), row[7]]) + " |")

    # NCU highlights
    lines += ["", "## NCU Highlights & Bottlenecks", ""]
    if ncu_map:
        for name,m in ncu_map.items():
            lines += [f"**{name}** — TC {fmt(m.get('tc_util_pct'))}% | Occ {fmt(m.get('achieved_occupancy_pct'))}% | DRAM {fmt(m.get('dram_bw_pct'))}% | Reg/thread {fmt(m.get('reg_per_thread'))}"]
    else:
        lines += ["(No NCU data yet. Run `bash scripts/profile.sh`.)"]

    # Autotune
    lines += ["", "## Autotune Summary", ""]
    if os.path.exists("artifacts/tune/tune_log.csv"):
        lines += ["See `artifacts/tune/tune_log.csv` for full log; `artifacts/tune/topk.json` lists the best configs."]
    else:
        lines += ["(No autotune results yet. Run `python scripts/evo_tune.py`.)"]

    # Risks/Next Actions
    lines += ["", "## Risks & Next Actions",
              "- Ensure correctness tolerances: atol=0.06, rtol=0.02.",
              "- If <15× vs Baseline A, prioritize: (1) tighten QK^T tiling; (2) increase stages/prefetch; (3) adjust num_warps for occupancy.",
              ""]

    Path("reports/summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/summary.md")

if __name__ == "__main__":
    main()
