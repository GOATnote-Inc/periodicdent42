#!/usr/bin/env python3
import json, pathlib, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
B = ROOT/"artifacts"/"bench"
N = ROOT/"artifacts"/"ncu"/"summary.json"
R = ROOT/"reports"; R.mkdir(parents=True, exist_ok=True)

def rd(p, k): return json.loads((B/p).read_text())[k]
def exists(p): return (B/p).exists()

def main():
    base_a = json.loads((B/"baseline_a.json").read_text())
    base_b = json.loads((B/"baseline_b.json").read_text())

    rows=[]
    for fn,tag in [("candidate_stage2.json","Stage2-Baseline"),
                   ("candidate_ws_p1.json","Stage5-WS-P1"),
                   ("candidate_ws_p2.json","Stage5-WS-P2")]:
        if not exists(fn): continue
        j = json.loads((B/fn).read_text())
        rows.append({
          "name": tag,
          "p50": j["p50_us"],
          "p90": j["p90_us"],
          "speedup_a": j["speedup_vs_baseline_a"],
          "speedup_b": j["speedup_vs_baseline_b"],
          "mean_abs_err": j["mean_abs_err"],
          "max_abs_err": j["max_abs_err"],
          "status": "PASS" if j["max_abs_err"] <= j.get("atol",6e-2) else "FAIL"
        })

    ncu = json.loads(N.read_text()) if N.exists() else {}
    def linem(name, key):
        # pick the closest ncu entry for the tag
        for k in ncu.keys():
            if name.lower().replace("-","_") in k: 
                val = ncu[k].get(key)
                if val is not None:
                    return f"{val:.1f}" if isinstance(val, (int,float)) else str(val)
        return "-"

    # Build markdown
    md = []
    md += [ "# SDPA — Mission Workload Summary",
            "",
            "## Top‑line numbers",
            "",
            "| Variant | p50 (μs) | p90 (μs) | Speedup× vs Baseline A | vs Baseline B | Mean abs err | Max abs err | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    md.append(f"| Baseline A (math) | {base_a['p50_us']:.2f} | {base_a['p90_us']:.2f} | 1.00× | {base_a['p50_us']/base_b['p50_us']:.2f}× | - | - | REF |")
    md.append(f"| Baseline B (flash) | {base_b['p50_us']:.2f} | {base_b['p90_us']:.2f} | {base_b['p50_us']/base_a['p50_us']:.2f}× | 1.00× | - | - | REF |")
    for r in rows:
        md.append(f"| {r['name']} | {r['p50']:.2f} | {r['p90']:.2f} | **{r['speedup_a']:.2f}×** | {r['speedup_b']:.2f}× | {r['mean_abs_err']:.4f} | {r['max_abs_err']:.4f} | {r['status']} |")

    md += ["", "## NCU highlights (best‑effort)",
           "",
           "| Variant | SM util % | TC util % | Ach. occ % | Eligible warps/cycle | L2 hit % | DRAM %peak |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        name = r["name"]
        md.append(f"| {name} | {linem(name,'sm_util_pct')} | {linem(name,'tensor_core_util_pct')} | {linem(name,'achieved_occupancy_pct')} | {linem(name,'eligible_warps_per_cycle')} | {linem(name,'l2_hit_rate_pct')} | {linem(name,'dram_bw_pct_of_peak')} |")

    # Risks + next actions (auto‑filled if <15×)
    achieved = max([r["speedup_a"] for r in rows]) if rows else 0.0
    status = "✅ **Target met (≥15× vs Baseline A)**" if achieved >= 15.0 else f"⚠️  **Target not met** (best {achieved:.2f}×)"
    md += ["", f"**Status:** {status}", ""]
    if achieved < 15.0:
        md += ["### Bottlenecks (from NCU, heuristic):",
               "- Top warp stalls by percentage (see artifacts/ncu/summary.json).",
               "- Tensor Core under‑utilization or DRAM %peak too high indicates memory‑bound.",
               "",
               "### Next 3 levers (expected impact if bottlenecked):",
               "1) Increase `NUM_STAGES` / `PREFETCH` depth to hide latency (**~5–15%**).",
               "2) Try `PROD_WARPS=1→2` or smaller `BLOCK_K` to balance register pressure vs overlap (**~3–10%**).",
               "3) Re‑layout KV to improve L2 locality (`KV_LAYOUT=interleaved`) (**~5–12%**).",
               "",
               "### Files Generated:",
               f"- `artifacts/bench/*.json` — Performance data (p50/p90/CI)",
               f"- `artifacts/ncu/*.ncu-rep` — Raw NCU profiles",
               f"- `artifacts/ncu/summary.json` — Parsed NCU metrics",
               f"- `artifacts/tune/tune_log.csv` — EvoEngineer-Full search log",
               f"- `artifacts/tune/topk.json` — Top-6 elites",
               f"- `artifacts/manifest.yaml` — Environment snapshot",
        ]
    else:
        md += ["### 🎉 Congratulations!",
               f"Your kernel achieved **{achieved:.2f}× speedup** vs PyTorch math baseline.",
               "",
               "### Files Generated:",
               f"- `artifacts/bench/*.json` — Performance data (p50/p90/CI)",
               f"- `artifacts/ncu/*.ncu-rep` — Raw NCU profiles",
               f"- `artifacts/ncu/summary.json` — Parsed NCU metrics",
               f"- `artifacts/tune/tune_log.csv` — EvoEngineer-Full search log",
               f"- `artifacts/tune/topk.json` — Top-6 elites",
               f"- `artifacts/manifest.yaml` — Environment snapshot",
        ]

    (R/"summary.md").write_text("\n".join(md))
    print(f"✅ Report written to {R/'summary.md'}")

if __name__=="__main__":
    main()
