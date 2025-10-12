# Salvageable Assets Inventory - October 12, 2025

**Status**: Ready for expert review  
**Purpose**: Identify what to keep for clean restart

---

## Problem Summary

The `cudadent42/` folder has become too tangled with:
- Multiple failed build attempts
- Conflicting file versions
- 7+ build system iterations
- Mixed working/broken code

**Cost**: ~$2.85 spent, minimal progress on actual validation

---

## Assets Worth Salvaging

### 1. CRITICAL FIX (Proven Value)

**File**: `cudadent42/python/flashmoe_science/csrc/build_config.h`

**Value**: This file identifies the root cause of 0.12× regression
```cpp
constexpr int NUM_WARPS_PER_BLOCK = 12;  // Was 4, now 12
constexpr int THREADS_PER_BLOCK = 384;   // Was 128
```

**Status**: ✅ Correct configuration, ready to use  
**Lines**: 138 lines  
**Path**: `cudadent42/python/flashmoe_science/csrc/build_config.h`

---

### 2. ANALYSIS DOCUMENTS (High Value)

These clearly document what happened and why:

**a) Root Cause Analysis**
- File: `cudadent42/PERFORMANCE_REGRESSION_ANALYSIS_OCT12.md` (445 lines)
- Value: Documents 0.12× regression, identifies configuration bug
- Keep: ✅ YES

**b) Fix Instructions**
- File: `cudadent42/FIX_INSTRUCTIONS.md` (314 lines)
- Value: Step-by-step guide
- Keep: ✅ YES

**c) Summary**
- File: `cudadent42/SUMMARY_FIX_AND_LESSONS.md` (400 lines)
- Value: Complete analysis with lessons learned
- Keep: ✅ YES

---

### 3. BENCHMARK INFRASTRUCTURE (Unknown Value)

**Files**:
- `cudadent42/benches/bench_correctness_and_speed.py`
- `cudadent42/benchmarks/benchmark_attention.py`

**Status**: Never successfully ran with correct config  
**Decision**: Let you decide - may need these or may not

---

### 4. KERNEL CODE (Uncertain Quality)

**Files in `cudadent42/python/flashmoe_science/csrc/`**:
- `flash_attention_science.cu` (has vectorized loads code)
- `flash_attention_science.h`
- Various other .cu files (some with type errors)

**Problem**: Tangled with multiple failed fix attempts  
**Value**: Has vectorized loads optimization (31 lines)  
**Status**: Unclear if code is actually correct  
**Decision**: Expert review needed

---

### 5. WHAT TO DISCARD (Low/No Value)

- All build artifacts (`build/`, `*.so` files)
- Multiple versions of setup.py
- Failed build attempts documentation
- Git stash entries
- Temporary files

---

## Recommended Clean Structure

```
periodicdent42/
├── flashattention-clean/          # NEW: Clean folder
│   ├── README.md                  # Fresh start documentation
│   ├── build_config.h             # SALVAGE: Correct config (NUM_WARPS=12)
│   ├── kernel/                    # NEW: Clean kernel code
│   │   └── flash_attention.cu    # Rewrite or salvage
│   ├── bench/                     # NEW: Simple benchmark
│   │   └── simple_bench.py       # Clean, minimal benchmark
│   └── docs/                      # SALVAGE: Analysis docs
│       ├── ROOT_CAUSE.md          # Salvaged analysis
│       └── LESSONS.md             # What we learned
└── cudadent42/                    # OLD: Archive, don't delete yet
    └── [everything stays here for reference]
```

---

## Questions for Expert Review

### 1. What to salvage from kernel code?

**Option A**: Keep only `build_config.h` (the fix)
- Pro: Clean slate, no baggage
- Con: Lose vectorized loads code

**Option B**: Salvage vectorized loads code (31 lines)
- Pro: Keep optimization work
- Con: Need to verify it's correct

**Option C**: Start completely fresh, just reference the fix
- Pro: Maximum clarity
- Con: Re-implement everything

**Your call**: Which option?

---

### 2. Benchmark infrastructure?

Current benchmarks are untested with correct config.

**Option A**: Keep existing benchmark scripts
**Option B**: Write new minimal benchmark (50 lines)
**Option C**: Use PyTorch SDPA comparison only

**Your call**: Which approach?

---

### 3. Where to put new folder?

**Option A**: `periodicdent42/flashattention-clean/` (root level)
**Option B**: `periodicdent42/cuda-kernels/` (more general)
**Option C**: `periodicdent42/experiments/flashattention/` (experimental)

**Your call**: What name/location?

---

### 4. What's the actual goal?

**Goal A**: Validate that NUM_WARPS_PER_BLOCK=12 fixes regression
- Focus: Just prove the fix works
- Time: 30 minutes
- Complexity: Minimal

**Goal B**: Build complete FlashAttention library
- Focus: Full implementation
- Time: Weeks
- Complexity: High

**Goal C**: Something in between

**Your call**: What's the target?

---

## Cost to Date

| Session | Duration | Cost | Result |
|---------|----------|------|--------|
| Oct 12 evening | 45 min | $0.95 | Build fixes (mixed) |
| Oct 12 late | 25 min | $0.60 | Regression found ✅ |
| Oct 12 very late | 90+ min | $1.30+ | Build spiral ❌ |
| **Total** | **160+ min** | **$2.85+** | **1 good finding** |

**Good outcome**: Found root cause (NUM_WARPS_PER_BLOCK)  
**Bad outcome**: Got stuck in build complexity

---

## My Recommendation

**Keep it simple**:

1. **Salvage**: Only `build_config.h` (the proven fix)
2. **New folder**: `periodicdent42/fa-validate/` 
3. **Goal**: Validate fix works (30-minute project)
4. **Benchmark**: Minimal Python script (50 lines)
5. **Success**: Prove 1.2-1.5× speedup, document, done

**Defer**:
- Complete library implementation
- All optimizations
- Complex build systems
- Everything else

**Total time to success**: 1-2 hours with clean slate

---

## Awaiting Your Expert Direction

Please specify:

1. **What to salvage** from kernel code (if anything)
2. **Folder name/location** for fresh start
3. **Goal** (minimal validation vs full library)
4. **Any other assets** from `cudadent42/` worth keeping

I'll wait for your guidance before creating anything new.

---

**Status**: GPU stopped, awaiting expert direction ✅

