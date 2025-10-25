# 🚀 Stage-5 L4 Execution Guide

**Status**: Ready for GPU testing  
**Date**: October 21, 2025  
**Target**: ≥15× speedup vs PyTorch SDPA (math backend)

---

## 📋 Prerequisites

### 1. SSH into L4 Instance

```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c
```

### 2. Navigate to Workspace

```bash
cd ~/periodicdent42
```

### 3. Pull Latest Code

```bash
git fetch origin
git checkout feat/stage5-warp-spec-persistent
git pull origin feat/stage5-warp-spec-persistent
```

### 4. Ensure Python Dependencies

```bash
# Should already be installed from previous sessions
pip install torch numpy pyyaml ninja
```

---

## 🎯 One-Command Execution

From the `sdpa_ws_pipeline/` directory:

```bash
cd sdpa_ws_pipeline
bash scripts/repro.sh
```

**That's it!** This single command will:

1. **Capture environment** (GPU, CUDA, PyTorch versions) → `artifacts/manifest.yaml`
2. **Benchmark baselines** (PyTorch SDPA math & flash) → `artifacts/bench/baseline_*.json`
3. **Benchmark 3 candidates**:
   - Stage-2 baseline (cp.async + WMMA P·V)
   - Stage-5 WS (NUM_PRODUCER_WARPS=1)
   - Stage-5 WS (NUM_PRODUCER_WARPS=2)
4. **Run EvoEngineer-Full autotune** (128 trials, top-6 elites) → `artifacts/tune/`
5. **Profile with NCU** (baseline + top-3 candidates) → `artifacts/ncu/*.ncu-rep`
6. **Generate summary report** → `reports/summary.md`

---

## ⏱️ Expected Runtime

| Phase | Duration | Notes |
|-------|----------|-------|
| Environment capture | ~5 sec | GPU info, versions |
| Baseline benchmarking | ~2 min | 100 iters × 2 backends |
| Candidate benchmarking | ~3 min | 100 iters × 3 variants |
| EvoEngineer-Full search | ~30-60 min | 128 trials, elite preservation |
| NCU profiling | ~10 min | 4 variants × 20 iters |
| Report generation | ~5 sec | Markdown synthesis |
| **Total** | **~45-75 min** | Depends on budget |

> **Tip**: To speed up for initial testing, set `BUDGET=32` (8-15 min for autotune):
> ```bash
> BUDGET=32 bash scripts/repro.sh
> ```

---

## 📊 Expected Results

### Success Criteria (Target: ≥15× vs Baseline A)

```
Baseline A (math):    ~1500-2000 μs (PyTorch reference)
Baseline B (flash):   ~100-150 μs (PyTorch optimized)
Stage-2 baseline:     ~650-700 μs (our control, 2-3× vs Baseline B)
Stage-5 WS-P1:        ~550-600 μs (target: +10-15% vs Stage-2)
Stage-5 WS-P2:        ~500-550 μs (target: +15-20% vs Stage-2)
```

**Target**: Stage-5 WS-P2 should achieve **≥15× vs Baseline A** (i.e., ≤100-130 μs).

---

## 📁 Artifacts Generated

After `repro.sh` completes, you'll have:

```
sdpa_ws_pipeline/
├── artifacts/
│   ├── bench/
│   │   ├── baseline_a.json         # PyTorch math backend
│   │   ├── baseline_b.json         # PyTorch flash backend
│   │   ├── candidate_stage2.json   # Our Stage-2 control
│   │   ├── candidate_ws_p1.json    # WS with 1 producer warp
│   │   └── candidate_ws_p2.json    # WS with 2 producer warps
│   ├── ncu/
│   │   ├── baseline.ncu-rep        # NCU profile (baseline)
│   │   ├── candidate_stage2.ncu-rep
│   │   ├── candidate_ws_p1.ncu-rep
│   │   ├── candidate_ws_p2.ncu-rep
│   │   └── summary.json            # Parsed NCU metrics
│   ├── tune/
│   │   ├── tune_log.csv            # All 128 trial configs + scores
│   │   └── topk.json               # Top-6 elite configs
│   └── manifest.yaml               # Environment snapshot
└── reports/
    └── summary.md                  # **THE FINAL REPORT** ✨
```

---

## 📖 Reading the Report

Open `reports/summary.md` to see:

### 1. Top-Line Table

| Variant | p50 (μs) | Speedup× vs A | vs B | Max abs err | Status |
|---------|----------|---------------|------|-------------|--------|
| Baseline A | 1500 | 1.0× | 15.0× | - | REF |
| Baseline B | 100 | 15.0× | 1.0× | - | REF |
| Stage2-Baseline | 650 | 2.3× | 0.15× | 0.045 | PASS |
| Stage5-WS-P1 | 580 | 2.6× | 0.17× | 0.048 | PASS |
| Stage5-WS-P2 | 530 | 2.8× | 0.19× | 0.051 | PASS |

### 2. NCU Highlights

| Variant | SM util % | TC util % | Occupancy % | DRAM %peak |
|---------|-----------|-----------|-------------|------------|
| Stage5-WS-P2 | 85.2 | 67.3 | 75.0 | 45.6 |

### 3. Status & Next Actions

- **✅ Target met**: If best ≥15× vs Baseline A
- **⚠️ Target not met**: Lists top bottlenecks + 3 recommended levers

---

## 🔍 Debugging Individual Steps

If `repro.sh` fails, run stages individually:

### Step 0: Environment

```bash
python3 scripts/capture_env.py
cat artifacts/manifest.yaml  # Verify GPU = L4, CUDA 12.x
```

### Step 1: Benchmarking

```bash
bash scripts/bench.sh
cat artifacts/bench/baseline_a.json  # Should show p50 in μs
```

### Step 2: Autotune (Fast Version)

```bash
BUDGET=32 python3 scripts/evo_tune.py --shape 2,8,512,64
cat artifacts/tune/topk.json  # Top-6 configs
```

### Step 3: Profiling

```bash
bash scripts/profile.sh
ls artifacts/ncu/*.ncu-rep  # Should have 4 .ncu-rep files
```

### Step 4: Report

```bash
python3 scripts/summarize.py
cat reports/summary.md
```

---

## 🚨 Troubleshooting

### Issue 1: `ncu` Permission Denied

**Solution**: NCU requires `sudo`. The script already uses `sudo ncu`, but if it prompts for password, you may need to:

```bash
sudo -v  # Refresh sudo timestamp
bash scripts/profile.sh
```

### Issue 2: Build Errors ("nvcc not found")

**Solution**: Ensure CUDA is in PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
```

### Issue 3: `ImportError: No module named 'tasks.fp8_sdpa_stage_c_wmma'`

**Solution**: You're not in the repo root. Make sure:

```bash
cd ~/periodicdent42/sdpa_ws_pipeline
python3 scripts/kbench.py --help  # Should work
```

### Issue 4: Correctness Failures (max_abs_err > 0.06)

**Interpretation**: FP8 quantization noise. The harness uses `atol=0.06` by default. Check:

- `artifacts/bench/candidate_*.json` → `"correctness_pass": true/false`
- If `false`, inspect `mean_abs_err` and `max_abs_err` to diagnose

---

## 📤 Sharing Results

After `repro.sh` completes:

### Option 1: Copy Report to Local Machine

```bash
# On L4
cd ~/periodicdent42/sdpa_ws_pipeline
tar -czf stage5_results.tar.gz artifacts/ reports/

# On local Mac
gcloud compute scp cudadent42-l4-dev:~/periodicdent42/sdpa_ws_pipeline/stage5_results.tar.gz . --zone=us-west1-c
tar -xzf stage5_results.tar.gz
open reports/summary.md
```

### Option 2: Commit & Push Artifacts

```bash
cd ~/periodicdent42
git add sdpa_ws_pipeline/artifacts sdpa_ws_pipeline/reports
git commit -m "perf(stage5): L4 validation results — [X.XX]× vs baseline A"
git push origin feat/stage5-warp-spec-persistent
```

### Option 3: View Report on L4

```bash
cat reports/summary.md | less
```

---

## ✅ Success Criteria Checklist

- [ ] `artifacts/manifest.yaml` shows GPU = "NVIDIA L4"
- [ ] `baseline_a.json` shows p50 > 1000 μs (slow math backend)
- [ ] `baseline_b.json` shows p50 < 200 μs (fast flash backend)
- [ ] `candidate_stage2.json` shows `correctness_pass: true`
- [ ] `candidate_ws_p1.json` shows `correctness_pass: true`
- [ ] `candidate_ws_p2.json` shows `correctness_pass: true`
- [ ] Best speedup ≥ 15× vs baseline A (target met!)
- [ ] `reports/summary.md` shows "✅ Target met"

---

## 🎯 Next Steps After Validation

### If Target Met (≥15×)

1. **Merge to main**:
   ```bash
   git checkout main
   git merge feat/stage5-warp-spec-persistent
   git tag v5.0-stage5-warp-spec
   git push origin main --tags
   ```

2. **Update STATUS_CURRENT.md** with final performance numbers

3. **Create final session report**: `SESSION_STAGE5_COMPLETE_OCT21_2025.md`

### If Target Not Met (<15×)

1. **Inspect NCU bottlenecks** in `artifacts/ncu/summary.json`
2. **Try recommended levers** from `reports/summary.md`
3. **Run focused autotune** on promising config space regions
4. **Document as "valid negative result"** if no further wins found

---

## 📞 Contact

If you encounter issues not covered here, check:

1. `sdpa_ws_pipeline/INTEGRATION_GUIDE.md` — Kernel integration details
2. `docs/STAGE5_PLAN.md` — Original implementation plan
3. `docs/WS_IMPLEMENTATION_GUIDE.md` — WS kernel internals

---

**🚀 You're ready! Just run:**

```bash
cd ~/periodicdent42/sdpa_ws_pipeline
bash scripts/repro.sh
```

**Good luck! 🎉**

