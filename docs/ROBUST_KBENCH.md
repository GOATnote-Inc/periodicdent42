# Robust Kernel Benchmarking Methodology

**Aligned with EvoEngineer Evaluation (Sec. 4.3, 5.1)**

---

## 🎯 Goals

1. **Reproducible** results across runs and machines
2. **Statistical** significance via median of many samples
3. **Modular** evaluation: separate compile, correctness, and performance gates
4. **Transparent** comparison to PyTorch baseline

---

## 📊 Methodology

### 1. Sample Size
- **Performance**: 100 iterations minimum
- **Warmup**: 20 iterations (discard results)
- **PyTorch baseline**: 50 iterations (slower, so fewer samples)

**Rationale**: 100+ samples provide stable median estimates. EvoEngineer uses ≥100 runs for statistical validity (Sec. 5.1).

### 2. Timing Mechanism
- **CUDA Events** (`torch.cuda.Event`) for GPU timing
- Synchronize before and after kernel launch
- Convert to microseconds for readability

**Why not CPU time?**: CPU timers include host overhead and async launch latency.

### 3. Reporting Metrics
- **p50 (median)**: Primary metric (robust to outliers)
- **p90**: 90th percentile (captures tail latency)
- **p99**: 99th percentile (worst-case performance)
- **mean ± std**: For trend analysis

**Rationale**: Median is more robust than mean for skewed distributions (cold cache, thermal throttling).

### 4. Correctness Validation
- **Reference**: PyTorch SDPA (flash backend, FP16)
- **Tolerance**: `max_err ≤ 0.06` (FP8-appropriate)
- **Additional gates**:
  - `mean_err ≤ 0.02` (per-element average error)
  - `%bad ≤ 1.0%` (% of elements exceeding tolerance)

**Why 0.06?**: FP8 quantization introduces ~0.03-0.05 noise; 0.06 provides headroom.

### 5. Random Seeds
- **Fixed seed** (default 0) for reproducibility
- **Multiple seeds** (0, 1, 2) for robustness testing

**Rationale**: Fixed seeds ensure bit-exact reproducibility; multiple seeds catch input-dependent bugs.

---

## 🚪 Modular Evaluation Gates

### Gate 1: Compile
```python
ext = build_extension()
# Check PTXAS output: regs ≤ 120, SMEM ≤ 64 KB, spills = 0
```

**Exit if fails**: No point testing broken code.

### Gate 2: Correctness
```python
ref = pytorch_sdpa(Q, K, V, scale)
out = our_kernel(Q, K, V, scale)
max_err = (out - ref).abs().max()
assert max_err <= 0.06, "Correctness gate failed"
```

**Exit if fails**: Performance means nothing if results are wrong.

### Gate 3: Performance
```python
times = [time_one_iter() for _ in range(100)]
p50 = median(times)
assert p50 <= target, "Performance gate failed"
```

**Only run if Gates 1 & 2 pass**: Save time by failing fast.

---

## 📋 Standard Shape Configurations

| Shape | B | H | S | D | Description |
|-------|---|---|-----|---|----|
| `small` | 1 | 8 | 128 | 64 | Quick correctness check |
| `mission` | 2 | 8 | 512 | 64 | Primary optimization target (L4) |
| `long` | 2 | 8 | 2048 | 64 | Long-sequence validation |

**Why these shapes?**
- `small`: Fast iteration for debugging
- `mission`: Representative of real workloads (512 tokens)
- `long`: Stress-test memory subsystem (2K tokens)

---

## 🔄 Comparison to PyTorch SDPA

### Why Compare?
- **Baseline**: PyTorch SDPA is production-quality (used in LLaMA, GPT, etc.)
- **Validation**: If we're slower than PyTorch, something is wrong
- **Marketing**: "X× faster than PyTorch" is a concrete claim

### Fair Comparison
- **Same precision**: PyTorch FP16 vs our FP8 (account for quantization)
- **Same backend**: PyTorch flash attention (not math backend)
- **Same device**: L4 GPU, same CUDA version
- **Same input**: Identical Q, K, V tensors (before/after quantization)

### Expected Speedups
- **vs PyTorch FP16**: 10-20× (FP8 is ~2× faster, plus our optimizations)
- **vs PyTorch FP8** (when available): 2-5× (algorithmic wins)

---

## 🧪 Example Output

```
==============================================================================
Gate 1: COMPILE
==============================================================================
✅ Extension built successfully

==============================================================================
Gate 2: CORRECTNESS + PERFORMANCE
==============================================================================
  [small   ] PyTorch baseline... 1234.56 μs
  [small   ] Warmup... done
  [small   ] Correctness... max_err=0.0459, mean_err=0.0142, %bad=0.0% → PASS
  [small   ] Timing 100 iters... p50=67.42 μs (18.3× vs PyTorch)
  [mission ] PyTorch baseline... 4567.89 μs
  [mission ] Warmup... done
  [mission ] Correctness... max_err=0.0532, mean_err=0.0178, %bad=0.0% → PASS
  [mission ] Timing 100 iters... p50=298.45 μs (15.3× vs PyTorch)
  [long    ] PyTorch baseline... 18234.56 μs
  [long    ] Warmup... done
  [long    ] Correctness... max_err=0.0598, mean_err=0.0189, %bad=0.1% → PASS
  [long    ] Timing 100 iters... p50=1124.67 μs (16.2× vs PyTorch)

==============================================================================
RESULTS SUMMARY
==============================================================================
small   : p50=  67.42μs  speedup= 18.3×  max_err=0.0459  ✅ PASS
mission : p50= 298.45μs  speedup= 15.3×  max_err=0.0532  ✅ PASS
long    : p50=1124.67μs  speedup= 16.2×  max_err=0.0598  ✅ PASS

📁 Results saved to: kbench/results_stage5.json

✅ ALL GATES PASSED
```

---

## 📁 Output Format (JSON)

```json
[
  {
    "shape": "mission",
    "B": 2,
    "H": 8,
    "S": 512,
    "D": 64,
    "seed": 0,
    "p50_us": 298.45,
    "p90_us": 312.34,
    "p99_us": 345.67,
    "mean_us": 301.23,
    "std_us": 12.45,
    "torch_p50_us": 4567.89,
    "speedup_vs_torch": 15.3,
    "max_err": 0.0532,
    "mean_err": 0.0178,
    "bad_pct": 0.0,
    "tol": 0.06,
    "correctness_pass": true
  }
]
```

**Why JSON?**
- Machine-readable for analysis scripts
- Version-controllable for regression tracking
- Easy to compare across commits

---

## 🔍 Debugging Failed Tests

### Correctness Failure
```bash
# Enable debug prints
CXXFLAGS="-DDEBUG_PRINT=1" python -m tasks.fp8_sdpa_stage_c_wmma.build
python scripts/bench_sdpa.py --shapes small --iters 1
```

### Performance Regression
```bash
# Compare against baseline
git checkout main
python scripts/bench_sdpa.py --out kbench/baseline.json

git checkout feat/stage5-warp-spec-persistent
python scripts/bench_sdpa.py --out kbench/stage5.json

# Analyze diff
python -c "
import json
b = json.load(open('kbench/baseline.json'))
s = json.load(open('kbench/stage5.json'))
for b_r, s_r in zip(b, s):
    pct = 100 * (s_r['p50_us'] - b_r['p50_us']) / b_r['p50_us']
    print(f\"{b_r['shape']}: {b_r['p50_us']:.2f} → {s_r['p50_us']:.2f} ({pct:+.1f}%)\")
"
```

---

## 📖 References

- **EvoEngineer Sec. 5.1**: Median-based reporting for statistical validity
- **FlashAttention Benchmark**: 100+ runs, p50/p90/p99 metrics
- **CUDA Best Practices**: Use CUDA events for accurate GPU timing

---

**Status**: Methodology documented ✅  
**Next**: Apply to Stage-5 validation (`scripts/bench_sdpa.py`)

