# Stage-5: Next Steps (GPU Required)

**Status**: ✅ **Implementation Complete**, ⏳ **Validation Pending**  
**Branch**: `feat/stage5-warp-spec-persistent`  
**Last Commit**: `d9604a0`

---

## 🎯 What's Done

✅ **Phase 1**: Infrastructure (benchmarking, NCU, autotune, docs)  
✅ **Phase 2**: WS kernel implementation (producer/consumer split)  
✅ **L4 Validation Script**: `scripts/run_stage5_validation_l4.sh` (comprehensive, one-click)

---

## 🚀 What's Next (GPU Required)

### Option A: Run Full Validation (Recommended, 2-4 hours)

```bash
# 1. SSH to L4
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c

# 2. Pull latest
cd ~/periodicdent42
git fetch -p && git checkout feat/stage5-warp-spec-persistent && git pull

# 3. Run validation script
bash scripts/run_stage5_validation_l4.sh
```

**The script will**:
- Build Stage-2 control + WS variants (NUM_PRODUCER_WARPS={1,2})
- Capture PTXAS stats (regs, SMEM, spills)
- Run 100-iteration benchmarks with PyTorch comparison
- Compare results (Stage-2 vs WS)
- (Optional) NCU profiling
- (Optional) EvoEngineer-Full autotune (elite K=3)
- Package artifacts (`kbench/*.json`, logs, ENV.json)

**Expected output**:
```
Variant         | Shape    | p50 Latency | vs PyTorch | max_err | Status
----------------|----------|-------------|------------|---------|--------
Stage-2         | mission  |   656.00μs  |   16.0×    | 0.0532  | ✅ PASS
WS (P=1)        | mission  |   590.00μs  |   18.0×    | 0.0545  | ✅ PASS  ← TARGET

WS (P=1) vs Stage-2 (mission): +11.0%  ← ✅ PASS (≥+10%)
```

---

### Option B: Quick Smoke Test (10 minutes)

```bash
# On L4
cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH

# Build Stage-2
USE_CP_ASYNC=1 USE_WMMA_PV=1 python -m tasks.fp8_sdpa_stage_c_wmma.build

# Build WS
USE_WARP_SPECIALIZATION=1 NUM_PRODUCER_WARPS=1 python -m tasks.fp8_sdpa_stage_c_wmma.build

# Quick test (10 iters)
python scripts/bench_sdpa.py --shapes small --iters 10 --warmup 3
```

---

## ✅ Success Criteria

### Hard Gates (Must Pass)
1. **PTXAS**: ≤120 regs, ≤64 KB SMEM, 0 spills
2. **Correctness**: max_err ≤ 0.06, mean_err ≤ 0.02, %bad ≤ 1.0%
3. **Performance (mission)**: p50 ≤ 590 μs (≥+10% vs Stage-2 @ 656 μs)
4. **NCU**: TC ≥50% OR DRAM <50% peak

### If All Gates Pass ✅
```bash
# Commit artifacts
git add kbench/
git commit -m "feat(stage5): WS validation results — all gates PASS"
git push

# Merge to main
git checkout main
git merge feat/stage5-warp-spec-persistent
git tag v3.0-stage5-warp-spec
git push origin main --tags

# Update STATUS_CURRENT.md
```

### If Any Gate Fails ❌
- Review `SESSION_STAGE5_WS_IMPLEMENTATION_COMPLETE_OCT21_2025.md` → **Debugging Playbook**
- Check logs: `kbench/logs/*.txt`
- Consider:
  - Reduce `NUM_PRODUCER_WARPS` to 1 (if spills)
  - Increase `max_err` tolerance (if FP8 noise)
  - Document as valid negative (if performance regresses)

---

## 📁 Key Files

### Scripts
- `scripts/run_stage5_validation_l4.sh` — One-click validation (all steps)
- `scripts/bench_sdpa.py` — Robust benchmarking (100-run medians)
- `scripts/ncu_sdpa.sh` — NCU profiling
- `kbench/autotune_evo_full.py` — EvoEngineer-Full search

### Documentation
- `SESSION_STAGE5_WS_IMPLEMENTATION_COMPLETE_OCT21_2025.md` — Complete implementation summary
- `docs/STAGE5_PLAN.md` — Original implementation plan
- `docs/WS_IMPLEMENTATION_GUIDE.md` — Kernel implementation guide
- `docs/ROBUST_KBENCH.md` — Benchmarking methodology

### Kernel
- `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu` — WS implementation
- `tasks/fp8_sdpa_stage_c_wmma/build.py` — Build system with Stage-5 toggles

---

## 🎯 Decision Tree

```
START: Run scripts/run_stage5_validation_l4.sh
  ├─ PTXAS passes? (≤120 regs, ≤64 KB SMEM, 0 spills)
  │    ├─ YES → Continue
  │    └─ NO → Reduce NUM_PRODUCER_WARPS, rebuild, retry
  │
  ├─ Correctness passes? (max_err ≤ 0.06)
  │    ├─ YES → Continue
  │    └─ NO → Debug (see playbook), fix, retry
  │
  ├─ Performance gate? (≥+10% vs Stage-2)
  │    ├─ YES → ✅ SUCCESS! Merge to main, tag
  │    └─ NO → Valid negative, document, revert to Stage-2
  │
  └─ PyTorch speedup? (≥15×)
       ├─ YES → ⭐ EXCELLENT! Highlight in README
       └─ NO → ⚠️  Marginal, but acceptable if other gates pass
```

---

## 📊 Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| SSH + setup | 5 min | One-time |
| Build variants | 10 min | Stage-2 + WS P=1, P=2 |
| Benchmarks (3 variants × 3 shapes) | 30-60 min | 100 iters each |
| NCU profiling (optional) | 15 min | Requires sudo |
| Autotune (optional) | 2-4 hours | Elite K=3, 16 configs |
| **Total (w/o autotune)** | **1-2 hours** | Core validation |
| **Total (w/ autotune)** | **3-6 hours** | Full search |

---

## 🎉 Why This Matters

**If successful (≥+10%)**: First time we've beaten Stage-2 with a compute-side optimization!  
- Stage-1 (cp.async): +13.8%  
- Stage-2 (WMMA P·V): +83%  
- **Stage-5 (WS)**: +10-20% (TARGET)

**Cumulative speedup**: 8.0× → 8.8-9.6× (approaching 10× milestone!)

**vs PyTorch SDPA**: ~16× → ~18-20× (exceeds ≥15× gate)

---

## 🔗 Quick Links

- **L4 Instance**: `cudadent42-l4-dev` (us-west1-c)
- **Branch**: `feat/stage5-warp-spec-persistent`
- **Main Script**: `scripts/run_stage5_validation_l4.sh`
- **Results Dir**: `kbench/` (will be created)

---

## 💡 Pro Tips

1. **Run in tmux**: `tmux new -s stage5` (survives SSH disconnects)
2. **Check logs**: `tail -f kbench/logs/bench_ws_p1.txt` (live progress)
3. **Save early**: Commit after each major step (build, bench, NCU)
4. **If stuck**: Review debugging playbook in session summary

---

**Ready to go!** Just run `bash scripts/run_stage5_validation_l4.sh` on L4. 🚀

