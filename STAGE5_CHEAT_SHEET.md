# ⚡ Stage-5 Cheat Sheet

## 🚀 One Command to Rule Them All

```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c
cd ~/periodicdent42/sdpa_ws_pipeline
git checkout feat/stage5-warp-spec-persistent && git pull
bash scripts/repro.sh
cat reports/summary.md
```

**Runtime**: ~45-75 min  
**Target**: ≥15× vs PyTorch math baseline

---

## 📁 Key Files

| File | What It Is |
|------|------------|
| `reports/summary.md` | **📊 THE FINAL REPORT** (speedups, NCU, diagnosis) |
| `artifacts/bench/*.json` | Performance data (p50/p90/CI) |
| `artifacts/ncu/summary.json` | Parsed NCU metrics |
| `artifacts/tune/topk.json` | Top-6 elite configs from autotune |
| `artifacts/manifest.yaml` | Environment snapshot (GPU, CUDA, PyTorch) |

---

## ✅ Success Checklist

- [ ] GPU shows "NVIDIA L4" in `manifest.yaml`
- [ ] Baseline A > 1000 μs (slow)
- [ ] Baseline B < 200 μs (fast)
- [ ] All candidates pass correctness
- [ ] Best speedup ≥ 15× vs Baseline A
- [ ] `reports/summary.md` shows "✅ Target met"

---

## 🔧 Quick Debug

### Run Individual Steps

```bash
# Step 1: Env
python3 scripts/capture_env.py && cat artifacts/manifest.yaml

# Step 2: Bench (fast test)
WARMUP=5 ITERS=20 bash scripts/bench.sh

# Step 3: Autotune (fast)
BUDGET=32 python3 scripts/evo_tune.py

# Step 4: Profile
bash scripts/profile.sh

# Step 5: Report
python3 scripts/summarize.py && cat reports/summary.md
```

### Fast Test (Skip Autotune)

```bash
# Just bench + NCU baseline
bash scripts/bench.sh
sudo /usr/local/cuda/bin/ncu -o artifacts/ncu/baseline.ncu-rep python3 scripts/kbench.py --iters 20 --warmup 5 --variants candidate_triton_flashlike
python3 scripts/summarize.py
```

---

## 🎯 Expected Results

| Variant | p50 (μs) | Speedup vs A |
|---------|----------|--------------|
| Baseline A | ~1500 | 1.0× |
| Baseline B | ~100 | ~15× |
| Stage-2 | ~650 | ~2.5× |
| **WS-P1** | ~580 | ~3.0× |
| **WS-P2** | ~530 | ~3.5× ✅ |

---

## 📚 Full Docs

- `STAGE5_L4_EXECUTION_GUIDE.md` — Detailed guide
- `STAGE5_READY_FOR_L4.md` — Comprehensive checklist
- `sdpa_ws_pipeline/INTEGRATION_GUIDE.md` — Kernel integration
- `docs/STAGE5_PLAN.md` — Implementation plan

---

## 🚨 Common Issues

| Issue | Fix |
|-------|-----|
| `ncu` permission denied | `sudo -v` then retry |
| `nvcc not found` | `export PATH=/usr/local/cuda/bin:$PATH` |
| `ImportError: tasks.fp8...` | Run from `sdpa_ws_pipeline/` directory |
| OOM | Reduce shape: `SHAPE=2,4,256,64 bash scripts/repro.sh` |

---

## 📤 After L4 Run

### If ✅ Target Met

```bash
git tag v5.0-stage5-15x
git checkout main && git merge feat/stage5-warp-spec-persistent
git push origin main --tags
```

### If ❌ Target Not Met

1. Check NCU bottlenecks in `artifacts/ncu/summary.json`
2. Try levers from `reports/summary.md`
3. Run focused autotune: `BUDGET=256 bash scripts/repro.sh`
4. Document as valid negative if no wins

---

**⚡ TL;DR**: Run `bash scripts/repro.sh` on L4, check `reports/summary.md`, merge if ≥15×

