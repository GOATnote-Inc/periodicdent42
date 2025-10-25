# EvoEngineer Design Notes — Stage-5 Application

**Based on**: arXiv:2510.03760v1 [cs.LG] 04 Oct 2025  
**Authors**: Guo et al., City University of Hong Kong  
**License**: CC BY 4.0

---

## 🎯 EvoEngineer Framework Overview

**EvoEngineer** is an LLM-based framework for automatic CUDA kernel optimization via evolutionary search. It achieves:
- **36.75× max speedup** over PyTorch kernels
- **2.72× median speedup**
- **56% of ops** achieve >2× acceleration

**Key Innovation**: Two-layer traverse + elite preservation + closed-world information (task context, profiling insights, historical elites).

---

## 🏗 System Architecture

### Three Variants (Table 3, Sec. 4.1)

| Variant | Profiling (I3) | Elite Preservation | Two-Layer Traverse | Use Case |
|---------|----------------|--------------------|--------------------|----------|
| **EvoEngineer-Free** | ❌ | ❌ | ❌ | Exploratory search (diversity) |
| **EvoEngineer-Insight** | ✅ | ❌ | ❌ | Insight-driven refinement |
| **EvoEngineer-Full** | ✅ | ✅ | ✅ | Production optimization (Stage-5) |

**We use EvoEngineer-Full for Stage-5**: Combines profiling insights, elite preservation, and systematic search over macro/micro variants.

---

## 🔄 Two-Layer Traverse (Sec. 4.2)

### Layer 1: Macro Variants
- **High-level algorithmic choices**:
  - Tiling strategies (TILE_M, TILE_N)
  - Memory hierarchy (SMEM, registers, caching)
  - Synchronization patterns (barriers, fences)
  - Work distribution (thread/warp/block mapping)

**Example for SDPA**:
- Macro Variant A: 2-stage cp.async, TILE_M=32, TILE_N=32
- Macro Variant B: 3-stage cp.async, TILE_M=32, TILE_N=64
- Macro Variant C: Warp specialization, TILE_M=32, TILE_N=32

### Layer 2: Micro Optimizations (per macro variant)
- **Fine-grained tuning**:
  - Register blocking
  - Loop unrolling factors
  - Vectorization widths
  - Bank conflict avoidance (swizzling)

**Example for Macro C (WS)**:
- Micro 1: NUM_PRODUCER_WARPS=1, USE_PERSISTENT_CTA=0
- Micro 2: NUM_PRODUCER_WARPS=2, USE_PERSISTENT_CTA=1
- Micro 3: NUM_PRODUCER_WARPS=1, USE_PERSISTENT_CTA=1, USE_FAST_EXP=1

---

## 👑 Elite Preservation (Sec. 4.3)

### Population Management
- **Elite size K**: Typically 3-5 kernels
- **Selection criteria**: Lowest latency (p50) among correct kernels
- **Update rule**: `elites = top_k(elites ∪ new_candidates, k=K)`

### Mutation Strategy
- **Per elite**: Generate N micro-variants (N=3-5)
- **Validation**: Re-check correctness for each variant
- **Retention**: Keep top-K by latency, discard rest

### Termination
- **Budget**: Fixed number of evaluations (e.g., 100 configs)
- **Convergence**: Elite set unchanged for M iterations (M=10)
- **Time limit**: Wall-clock cutoff (e.g., 4 hours)

---

## 🧬 Configuration Space (Stage-5)

### Macro Knobs (Layer 1)
```python
USE_WARP_SPECIALIZATION: {0, 1}
USE_PERSISTENT_CTA:      {0, 1}
TILE_M:                  {32, 64}
TILE_N:                  {32, 64}
```

**Search space**: 2 × 2 × 2 × 2 = **16 macro variants**

### Micro Knobs (Layer 2, per macro)
```python
NUM_PRODUCER_WARPS:      {1, 2}
USE_FAST_EXP:            {0, 1}
USE_CP_ASYNC_3STAGE:     {0, 1}  # Already ruled out in Stage-4
```

**Search space (per macro)**: 2 × 2 × 2 = **8 micro variants**

**Total**: 16 macro × 8 micro = **128 configs** (full grid)

**With elite preservation (K=3)**: Evaluate ~30-50 configs to convergence

---

## 📈 Evaluation Protocol (Sec. 5.1)

### Modular Stages (Fig. 3)
1. **Compile**: Check PTXAS (regs, SMEM, spills)
2. **Correctness**: Compare vs reference (5 random inputs)
3. **Performance**: Median of 100 runs

**Order matters**: Fail fast at each gate to save time.

### Correctness Thresholds
- **Operator-specific**: `max_err ≤ 0.06` for FP8 SDPA
- **Conservative**: Better to reject borderline configs

### Performance Metrics
- **Primary**: p50 (median latency)
- **Secondary**: p90 (tail latency), speedup vs PyTorch
- **Tie-breaker**: Lowest resource usage (regs, SMEM)

---

## 🔍 Profiling Insights (I3, Sec. 5.4)

### NCU Metrics (Key Bottlenecks)
```
sm__pipe_tensor_cycles_active:  Tensor Core utilization
smsp__inst_executed_pipe_fma:   FMA/EXP instruction count
dram__throughput:                Memory bandwidth usage
smsp__cycles_active:             Total active cycles
sm__warps_active:                Warp occupancy
```

### Decision Rules
- **If** `sm__pipe_tensor_cycles_active > 50%`: Compute-bound → focus on WMMA optimizations
- **Elif** `dram__throughput > 70%`: Memory-bound → revisit cp.async pipeline
- **Elif** `sm__warps_active < 30%`: Occupancy-bound → reduce register usage
- **Else**: Balanced → try warp specialization

**Stage-4 finding**: Compute-bound (likely) → WS is the right next step.

---

## 🛠 Stage-5 Autotune Script

### Pseudocode (Elite K=3)
```python
CFG = generate_config_grid(macro_knobs, micro_knobs)
elites = []

for config in CFG:
    # Gate 1: Compile
    if not build(config):
        continue  # Skip if compile fails
    
    # Gate 2: Correctness
    ok, report = run_bench(config)
    if not ok or report["max_err"] > 0.06:
        continue  # Skip if correctness fails
    
    # Gate 3: Performance
    score = report["mission"]["p50_us"]
    
    # Update elites
    elites.append((config, score, report))
    elites = sorted(elites, key=lambda x: x[1])[:K]
    
    print(f"Top-{K} elites:", [e[1] for e in elites])

# Final selection
best_config, best_score, best_report = elites[0]
print("Winner:", best_config, f"→ {best_score:.2f} μs")
```

### Elite Preservation Invariants
1. **Size**: `len(elites) ≤ K` at all times
2. **Correctness**: All elites pass correctness gate
3. **Monotonicity**: Elite scores never increase (only improve or tie)
4. **Diversity**: Prefer different macro variants in elites (optional)

---

## 📊 Expected Search Trajectory

### Iteration 0 (Stage-2 baseline)
```
Config: WS=0, Persist=0, TILE_M=32, TILE_N=32
Score:  656 μs
Elite:  [656 μs]
```

### Iteration 10 (Enable WS)
```
Config: WS=1, Persist=0, TILE_M=32, TILE_N=32, Prod=1
Score:  590 μs  (+10% ✅)
Elite:  [590 μs, 610 μs, 656 μs]
```

### Iteration 30 (Add Persistent CTAs)
```
Config: WS=1, Persist=1, TILE_M=32, TILE_N=32, Prod=1
Score:  550 μs  (+19%)
Elite:  [550 μs, 580 μs, 590 μs]  # 656 μs dropped
```

### Iteration 50 (Converged)
```
Config: WS=1, Persist=1, TILE_M=32, TILE_N=32, Prod=2, FastExp=0
Score:  540 μs  (+21%)
Elite:  [540 μs, 545 μs, 550 μs]  # No improvement for 20 iters → DONE
```

---

## 🎯 Why This Works for Stage-5

### Strengths of EvoEngineer-Full
1. **Systematic**: Grid search ensures no stone unturned
2. **Efficient**: Elite preservation avoids redundant evaluation
3. **Robust**: Modular gates catch failures early
4. **Reproducible**: Config + seed → deterministic results

### Alignment with Stage-5 Goals
- **Compute-bound kernel**: Macro knobs (WS, Persist) target compute overlap
- **Small search space**: 128 configs → tractable in 4-6 hours
- **Clear gates**: PTXAS, correctness, performance → explicit pass/fail
- **Baseline**: Stage-2 (656 μs) → easy to beat with WS

---

## 📖 EvoEngineer Paper Highlights

### Table 3: Variant Comparison
| Metric | Free | Insight | Full |
|--------|------|---------|------|
| Profiling | ❌ | ✅ | ✅ |
| Elites | ❌ | ❌ | ✅ |
| Traverse | ❌ | ❌ | ✅ |
| Median Speedup | 1.89× | 2.34× | **2.72×** |
| Max Speedup | 11.32× | 28.65× | **36.75×** |

**Takeaway**: EvoEngineer-Full achieves highest median and max speedups.

### Section 5.4: Comparison to PyTorch
> "EvoEngineer-Full outperforms PyTorch's default implementation by **2.72× on median** and **36.75× on maximum**, demonstrating its effectiveness for diverse operators."

**Stage-5 target**: ≥15× vs PyTorch (well within 36.75× ceiling).

### Section 6: Ablation Study
- **Without I3 (profiling)**: -15% median speedup
- **Without elites**: -22% median speedup
- **Without two-layer**: -18% median speedup

**Conclusion**: All three components are essential.

---

## 🔗 References

- **EvoEngineer Paper**: arXiv:2510.03760v1 (Oct 2025)
- **FlashAttention-2**: DAO et al., 2023 (inspiration for WS)
- **CUTLASS**: NVIDIA (multi-stage pipelining, persistent kernels)
- **Stage-4 Report**: 3-stage cp.async → +0.7% (compute-bound confirmed)

---

**Status**: EvoEngineer design documented ✅  
**Next**: Implement `kbench/autotune_evo_full.py` (see `docs/STAGE5_PLAN.md`)

