# SDPA WS Pipeline (Harness + Scripts)

**One command:**

```bash
bash scripts/repro.sh
```

This runs:
1. Bench: `scripts/bench.sh` (baselines + candidates, p50/p90 & 95% CI)
2. Tuning: `scripts/evo_tune.py` (EvoEngineer‑Full; logs every trial)
3. Profiling: `scripts/profile.sh` (Nsight Compute key metrics, *.ncu-rep)
4. Summary: `scripts/summarize.py` → `reports/summary.md`

Artifacts in `artifacts/` and `reports/`.
