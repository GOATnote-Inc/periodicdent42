# ✅ Repository Ready for Kernel Optimization - Oct 16, 2025

## Executive Summary

Your repository is **NOW** properly configured for AI-assisted CUDA kernel optimization using Cursor best practices as of October 2025.

---

## What Was Fixed

### 1. Repository Cleanup ✅
**Problem**: 170 MD files cluttering root, breaking AI context  
**Solution**: Archived 165 files → 5 essential docs remain  
**Result**: Clean, navigable structure

```
Root (5 files):
├── README.md                           # Project overview
├── CODEMAP.md                          # Navigation guide
├── KERNEL_FOCUS.md                     # 🎯 Optimization guide
├── CONTRIBUTING.md                     # Dev guidelines
└── SESSION_CLEANUP_OCT16_2025.md      # Cleanup report

Archive: docs/archive/session_logs/ (248 files, 3.2 MB)
```

### 2. Cursor AI Context Optimization ✅
**Problem**: AI lacked L4-specific guidance, no clear focus  
**Solution**: Created `.cursor/rules/kernel_optimization.md`  
**Result**: AI now knows:
- L4 (sm_89) constraints (48 KB SMEM, FP16 TC accumulation)
- Edit one change at a time, verify correctness
- Use region markers, preserve comments
- Red flags: SMEM overflow, register pressure, performance regressions

### 3. `.cursorignore` Precision ✅
**Problem**: Excluded ALL `.txt` files (killed benchmark outputs)  
**Solution**: Exclude only `data/**/*.txt`, keep `artifacts/bench/*.txt`  
**Result**: AI can read benchmark results, profile outputs

### 4. Single Source of Truth ✅
**Problem**: Optimization context scattered across 248 archived files  
**Solution**: Created `KERNEL_FOCUS.md` (160 lines)  
**Result**: Immediate orientation:
- Current kernel: `fa_s512_v3.cu` @ 38.00 μs (21% faster than PyTorch)
- Known issues: SMEM overflow, bank conflicts, FP32 penalty
- 4-phase optimization strategy (SMEM → Vectorization → Tensor Cores → L2)
- Success metrics and quick start commands

### 5. Broken Links Fixed ✅
**Problem**: `CODEMAP.md` linked to archived `V3_CLEAN_SLATE_ROADMAP.md`  
**Solution**: Updated to link `KERNEL_FOCUS.md` and `docs/archive/`  
**Result**: All links work, clear navigation path

---

## Cursor Best Practices Applied (Oct 2025)

Based on web search and kernel optimization requirements:

### ✅ Context Management
- **Focused scope**: 9 CUDA kernels, clear primary target (`fa_s512_v3.cu`)
- **Archived history**: 248 session logs out of immediate context
- **One source of truth**: `KERNEL_FOCUS.md` for quick orientation

### ✅ AI Rules & Boundaries
- **Editable zones**: Only `cudadent42/bench/**`, `scripts/**`, `.cursor/**`
- **No-go zones**: `docs/` (read-only), `infra/` (production), `ext/` (submodules)
- **Output format**: Diffs + commands, not prose

### ✅ Memory Efficiency
- **`.cursorignore`**: Excludes build artifacts, deps, large data (but keeps benchmarks)
- **Region markers**: `BEGIN X` / `END X` in `fa_s512_v3.cu` for surgical edits
- **Modular rules**: Separate files for base rules, kernel rules, no-edit zones

### ✅ Optimization Strategy
- **One change at a time**: Test after EVERY edit
- **Correctness first**: `torch.allclose` must pass before optimizing further
- **Profiling gates**: Use Nsight Compute to validate improvements
- **Documented changes**: Update comments for each non-obvious optimization

---

## Performance Baseline (Verified)

| Metric | PyTorch SDPA | fa_s512_v3.cu (Current) | Target (Phase 3) |
|--------|--------------|-------------------------|------------------|
| **Latency (p50)** | 47.10 μs | **38.00 μs** ✅ | 15-25 μs |
| **vs PyTorch** | 1.0× | **1.24× faster** | 1.88-3.14× faster |
| **SMEM Usage** | N/A | ~41 KB | < 48 KB |
| **TC Utilization** | Unknown | 0% (no WMMA) | > 50% |

**Hardware**: NVIDIA L4 (Ada, sm_89) on `cudadent42-l4-dev` (STOPPED)  
**Config**: B=2, H=8, S=512, D=64 (FP16)

---

## Next Steps (From KERNEL_FOCUS.md)

### Phase 1: SMEM Optimization (Target: 32-35 μs)
```bash
# Start GPU
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a

# SSH to GPU
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Navigate and build
cd ~/periodicdent42/cudadent42/bench
python build_v3_release.py

# Benchmark
cd ../../scripts
python bench_v3_quick.py
```

**Changes to make**:
1. ✅ Verify `S_smem` uses `half` (not `float`)
2. Add XOR swizzling for K/V SMEM layout
3. Confirm SMEM < 48 KB with `ptxas --verbose`

### Phase 2: Vectorized I/O (Target: 28-32 μs)
- Use `uint4` for 128-bit loads (8×fp16)
- Ensure 16-byte alignment
- Coalesced global memory access

### Phase 3: Tensor Core Integration (Target: 15-25 μs)
- WMMA for Q·K^T with FP16 accumulation
- Warp tiling (2×2 tiles)
- Verify TC utilization in Nsight

### Phase 4: L2 Cache Persistence (Target: 12-18 μs)
- Pin K/V in L2 using `cudaStreamSetAttribute`
- Leverage 48 MB L2 cache

---

## Repository Health Check

```bash
✅ Root directory: 5 essential files
✅ Archive: 248 session logs organized
✅ Kernels: 9 CUDA files (fa_s512_v3.cu is primary)
✅ .cursorignore: Precise exclusions
✅ .cursor/rules: 3 files (base, kernel_optimization, no_edit zones)
✅ CODEMAP.md: Up-to-date, no broken links
✅ KERNEL_FOCUS.md: Single source of truth
✅ Git: Clean working tree, main up-to-date
✅ GPU: Stopped (save costs)
✅ CI: Passing (repo organization, cleanup validated)
```

---

## Files to Read First

1. **KERNEL_FOCUS.md** (this file) - Optimization roadmap
2. **cudadent42/bench/kernels/fa_s512_v3.cu** - Primary kernel
3. **.cursor/rules/kernel_optimization.md** - AI editing rules
4. **docs/archive/session_logs/L4_ROADMAP_RECONCILED.md** - L4 deep dive
5. **docs/archive/session_logs/ITER1_CRITICAL_FINDINGS.md** - What NOT to do

---

## Success Criteria

### Immediate (5 min)
- ✅ Can navigate repo without confusion
- ✅ Can find primary kernel (`fa_s512_v3.cu`)
- ✅ Can read optimization strategy (Phase 1-4)

### Short-term (1 day)
- Implement Phase 1 (SMEM optimization)
- Verify correctness (`torch.allclose`)
- Measure improvement (target: 32-35 μs)

### Medium-term (1 week)
- Complete Phases 1-2 (SMEM + Vectorization)
- Achieve 28-32 μs (1.47-1.68× faster than PyTorch)
- Document each optimization

### Long-term (2-4 weeks)
- Complete Phases 3-4 (Tensor Cores + L2)
- Achieve 15-25 μs (1.88-3.14× faster than PyTorch)
- Publication-ready benchmark results

---

## Lessons Learned (From Archives)

### What Worked ✅
- Starting with working kernel (fa_s512_v3.cu @ 38 μs)
- One change at a time with verification
- L4-specific constraints documented upfront

### What Failed ❌
- **fa_s512.cu**: Fundamentally broken (misaligned address errors)
- **WMMA from scratch**: Complex, hit SMEM limits, abandoned
- **Flash-Attn-2 installation**: SSH flakiness, compilation issues
- **Warp-cooperative experiments**: 200× slowdown (unguarded printf)

### Key Insights 💡
1. **SMEM is the bottleneck on L4** (48 KB, not 64 KB)
2. **FP16 accumulation is 2× faster** on Ada Tensor Cores
3. **Bank conflicts are catastrophic** with HEAD_DIM=64
4. **PyTorch SDPA is slow** (47 μs - unoptimized Flash Attention)
5. **Optimization beats from-scratch** (38 μs existing vs 321 μs broken baseline)

---

## Commands Reference

```bash
# Repository
git status                              # Check repo state
git log --oneline -5                    # Recent commits

# GPU
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Build & Benchmark (on GPU)
cd ~/periodicdent42/cudadent42/bench
python build_v3_release.py             # Compile kernel
cd ../../scripts
python bench_v3_quick.py               # Quick benchmark (< 60s)

# Profile (on GPU)
cd ~/periodicdent42
make profile                           # Nsight Compute
```

---

**Status**: ✅ **READY FOR KERNEL OPTIMIZATION**  
**Commit**: `be04ad3` - Optimize repo for kernel development  
**GPU**: STOPPED (save costs)  
**Next**: Start GPU, implement Phase 1 (SMEM optimization)  

**Let's build the fastest FlashAttention kernel on L4! 🚀**

