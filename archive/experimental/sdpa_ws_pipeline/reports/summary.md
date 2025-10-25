# SDPA Warp‑Specialization Pipeline — Report

> **Status:** Scripts and harness ready. Run `bash scripts/repro.sh` on an NVIDIA L4 (or similar) to generate *all* artifacts and this report.

## What you'll get after running:

- Nsight Compute reports: `artifacts/ncu/*.ncu-rep` + `artifacts/ncu/summary.json`
- EvoEngineer‑Full logs: `artifacts/tune/tune_log.csv`, `artifacts/tune/topk.json`
- Bench JSONs: `artifacts/bench/*.json`
- This report auto-refreshed with a comparison table and NCU highlights.

### Mission‑shaped workload
- **B=2, H=8, S=512, D=64** (non‑causal, dropout off)

### Gates (acceptance criteria)
- ≥15× speedup vs PyTorch SDPA (Baseline A) on mission shape
- NCU + EvoEngineer‑Full artifacts present
- `bash scripts/repro.sh` reproduces the table on same hw class

---

## Comparison Table (to be populated)
Run `bash scripts/repro.sh` — the table below will be auto‑filled by `scripts/summarize.py`.

| Variant | p50 (μs) | Speedup vs A | Speedup vs B | TC util % | Occupancy % | DRAM %peak | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline A |  |  |  |  |  |  |  |
| Baseline B |  |  |  |  |  |  |  |
| Cand #1    |  |  |  |  |  |  |  |
| Cand #2    |  |  |  |  |  |  |  |
| Cand #3    |  |  |  |  |  |  |  |

---

## NCU Highlights & Bottleneck Analysis
This will summarize SM %, TC %, occupancy, L2 hit rate, DRAM BW, warp stalls from `artifacts/ncu/summary.json`.
