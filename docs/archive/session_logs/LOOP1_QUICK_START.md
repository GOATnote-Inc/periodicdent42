# Loop 1: CUDA Kernel Iteration - Quick Start

**Real iteration. Real learning. Real improvements.**

---

## What is Loop 1?

Loop 1 is the **CUDA kernel iteration loop** - the actual engineering work of making kernels faster:

```
Profile → Identify Bottleneck → Fix → Measure → Learn → Repeat
```

**NOT** statistical validation or publication prep (that's Loop 2).

---

## System Architecture

```
cudadent42/bench/
├── kernels/
│   └── fa_s512.cu              # Tunable FA kernel for L4 (SM_89)
├── fa_s512_tunable.py          # Python interface (JIT compilation)
├── search_space.py             # Tunables + hard gates (CUDA doctrine)
├── candidate_kernel.py         # Evaluation wrapper
└── loop1_optuna.py             # Main optimization loop
```

---

## Key Features

### 1. **Tunable Kernel** (`fa_s512.cu`)
- ✅ cp.async double-buffering (STAGES: 2-4)
- ✅ mma.sync with ldmatrix (BLOCK_M/N/K tunable)
- ✅ SMEM swizzle for bank conflicts (on/off)
- ✅ half2 vectorized loads (on/off)
- ✅ Online softmax (streaming attention)
- ✅ Persistent blocks (NUM_WARPS: 4-8)

**9 tunables = 3,888 configurations**

### 2. **Hard Gates** (CUDA Doctrine)
```python
def hard_gates(meta):
    if not meta['coalesced']: return "bad_coalescing"
    if meta['bank_conflicts'] > 0: return "bank_conflicts"
    if meta['occupancy'] < 0.5 and not CP_ASYNC: return "low_occupancy"
    if meta['smem_bytes'] > 49152: return "smem_overflow"
    if meta['peak_mb'] > 20000: return "oom_risk"
    if meta['max_rel_err'] > 1e-2: return "numerics"
    return None  # Pass
```

### 3. **Optimization Strategy**
1. **LHS Seed** (20 configs) - Broad exploration
2. **Optuna TPE** (100 trials) - Focused exploitation  
3. **Confirmation** (N=100) - Bootstrap CIs

### 4. **Success Criteria**
- ≥10% speedup over PyTorch SDPA
- Non-overlapping 95% CIs
- Passes all hard gates
- Correct output (max rel error < 1%)

---

## Quick Start (30 min setup + 2 hour search)

### Step 1: Install Dependencies

```bash
# On GPU instance
pip install optuna torch numpy scipy
```

### Step 2: Test System Components

```bash
cd /home/kiteboard/periodicdent42

# Test 1: Search space
python3 cudadent42/bench/search_space.py
# Expected: Print 3,888 configs, test config passes gates

# Test 2: Tunable kernel (builds and runs)
python3 cudadent42/bench/fa_s512_tunable.py
# Expected: Build success, median latency ~0.3-0.5 ms

# Test 3: Candidate evaluation
python3 cudadent42/bench/candidate_kernel.py
# Expected: Full evaluation with correctness check
```

### Step 3: Run Loop 1

```bash
# Full 2-hour optimization
python3 cudadent42/bench/loop1_optuna.py

# Custom run (adjust baseline and budget)
python3 -c "
from cudadent42.bench.loop1_optuna import Loop1Optimizer

opt = Loop1Optimizer(
    baseline_ms=0.3226,      # From TF32-fixed measurement
    target_speedup=1.10,     # 10% improvement goal
    budget_minutes=120       # 2 hours
)
opt.run()
"
```

---

## What You'll See

### Phase 1: LHS Seed (20 configs, ~20 min)
```
[LHS] Testing config:
  BLOCK_M     : 128
  BLOCK_N     : 64
  NUM_WARPS   : 4
  ...
  → Latency: 0.4521 ms
  → Gates:   FAIL (low_occupancy_no_prefetch)

[LHS] Testing config:
  BLOCK_M     : 256
  BLOCK_N     : 128
  NUM_WARPS   : 8
  ...
  → Latency: 0.3102 ms
  → Gates:   PASS
  → Speedup: 1.040× (+4.0%)
  ✅ BEATS TARGET (0.2933 ms)
```

### Phase 2: Optuna TPE (~100 trials, ~90 min)
```
[OPTUNA] Testing config:
  BLOCK_M     : 192
  BLOCK_N     : 96
  ...
  → Latency: 0.2875 ms
  → Gates:   PASS
  → Speedup: 1.122× (+12.2%)
  ✅ BEATS TARGET
```

### Phase 3: Confirmation (N=100, ~10 min)
```
Best candidate from search:
  Median: 0.2875 ms
  Config: {BLOCK_M: 192, BLOCK_N: 96, ...}

Re-running with N=100 for statistical confidence...
  → Latency: 0.2881 ms
  → 95% CI: [0.2875, 0.2889]

FINAL RESULTS
======================================================================
Baseline:  0.3226 ms
Best:      0.2881 ms (95% CI: [0.2875, 0.2889])
Speedup:   1.120× (+12.0%)
Target:    1.10×
CIs Overlap: False
Significant: True

🎉 SUCCESS: Achieved target speedup!
```

---

## Understanding Results

### Success Case (≥10% win)
```json
{
  "success": true,
  "best_config": {
    "BLOCK_M": 192,
    "BLOCK_N": 96,
    "BLOCK_K": 64,
    "NUM_WARPS": 8,
    "STAGES": 3,
    "UNROLL": 2,
    "CP_ASYNC": 1,
    "SWIZZLE": 1,
    "HALF2": 1
  },
  "best_latency_ms": 0.2881,
  "ci_95": [0.2875, 0.2889],
  "speedup": 1.120,
  "improvement_pct": 12.0
}
```

**Next**: Profile with Nsight to understand WHY it's faster

### Negative Result (No win found)
```json
{
  "success": false,
  "best_latency_ms": "inf",
  "speedup": 0.0,
  "candidates_found": 0,
  "rejected_count": 120
}
```

**Next**: Analyze rejections, pivot to different workload

---

## Common Rejection Reasons

| Reason | Meaning | Fix |
|--------|---------|-----|
| `bad_coalescing` | Memory access not coalesced | Enable HALF2=1 |
| `bank_conflicts` | SMEM bank conflicts | Enable SWIZZLE=1 |
| `low_occupancy_no_prefetch` | <50% occupancy, no cp.async | Enable CP_ASYNC=1 or increase warps |
| `smem_overflow` | >48KB shared memory | Reduce BLOCK_M/N or STAGES |
| `numerics` | Max rel error >1% | Check for NaN/Inf, reduce tile sizes |
| `build_failed` | CUDA compilation error | Check NVCC flags, architecture |

---

## Learning from the Loop

### What Loop 1 Teaches You (Even Without Wins)

1. **Which tile sizes** work on L4
2. **Whether cp.async** helps at S=512
3. **Optimal warp count** for this workload
4. **Bank conflict impact** (SWIZZLE on/off)
5. **Occupancy sweet spot** for SM_89

**This is science** - profile-backed evidence of what works.

---

## Next Steps After Loop 1

### If Success (≥10% win):
```bash
# Profile best config
ncu --set full --target-processes all \
    -o artifacts/profile_best \
    python3 -c "from cudadent42.bench.candidate_kernel import candidate_kernel; 
                candidate_kernel({...best_config...}, iterations=1)"

# Generate report
ncu --csv --page raw artifacts/profile_best.ncu-rep > profile_best.csv
```

### If Negative Result:
```bash
# Option A: Profile baseline to understand limits
ncu --set full python3 benchmark_sdpa.py

# Option B: Pivot to different workload
# - Decode path (causal + KV cache)
# - Long sequences (S≥2048)
# - Quantized KV (FP8/INT8)

# Option C: Multi-shape optimization
# - S=128: 4.6× faster (already proven)
# - S=256: 3.1× faster (already proven)
```

---

## Cost Analysis

| Phase | Duration | GPU Cost (L4 @ $0.68/hr) | Value |
|-------|----------|--------------------------|-------|
| Setup & Test | 30 min | $0.34 | System validation |
| LHS Seed | 20 min | $0.23 | Broad exploration |
| Optuna TPE | 90 min | $1.02 | Focused search |
| Confirmation | 10 min | $0.11 | Statistical proof |
| **Total** | **2.5 hours** | **$1.70** | **Real kernel iteration** |

**vs** Manual tuning: 8+ hours, $5.44, no systematic exploration

---

## Troubleshooting

### Build Errors
```bash
# Check CUDA version
nvcc --version  # Should be 12.x

# Check architecture
python3 -c "import torch; print(torch.cuda.get_device_capability())"
# Should be (8, 9) for L4
```

### OOM Errors
```bash
# Reduce STAGES or BLOCK_M/N
# Check: cudadent42/bench/search_space.py -> calculate_smem_usage()
```

### Slow Compilation
```bash
# Use cache (default: /tmp/fa_s512_cache)
# Subsequent builds reuse compiled kernels
```

---

## Files Generated

```
cudadent42/bench/artifacts/loop1/
├── loop1_results.json          # Final summary
└── (future) nsight_profiles/   # ncu reports for winners
```

---

## Remember

**Loop 1 = Iteration**
- Goal: Make kernel faster
- Output: Speedup + learning
- Metric: Latency (ms)

**Loop 2 = Validation**
- Goal: Prove improvement is real
- Output: Statistical confidence
- Metric: p-values, CIs, effect sizes

**Science = Loop 1 + Loop 2**

---

**Ready to iterate?** 🚀

```bash
python3 cudadent42/bench/loop1_optuna.py
```

