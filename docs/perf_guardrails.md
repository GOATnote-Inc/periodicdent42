# Performance Guardrails

**Purpose**: Enforce performance standards for custom CUDA kernels via automated CI gates.

---

## Quick Reference

### Pass Criteria

✅ **PASS** if:
- No regression > 3% (with non-overlapping 95% CIs)
- Correctness fuzzing: 100% pass rate
- Claimed improvements: ≥ 10% speedup AND non-overlapping CIs AND Cliff's δ ≥ 0.3

⚠️ **WARNING** if:
- CIs overlap (no significant difference)
- Effect size too small (Cliff's δ < 0.3)

❌ **FAIL** if:
- Regression > 3% with non-overlapping CIs
- Correctness fuzzing: any test fails
- Claimed improvement but CIs overlap or effect size < 0.3

---

## Correctness Fuzzing

**Script**: `cudadent42/bench/correctness_fuzz.py`

**Test Matrix**:
- S ∈ {448, 512, 640} (jittered sequence lengths)
- B ∈ {16, 32, 48} (varied batch sizes)
- H ∈ {4, 8, 16} (head counts)
- D = 64 (fixed head dimension)

**Tolerances** (FP16):
- Absolute: 2e-3
- Relative: 1e-3

**Oracle**: PyTorch SDPA (FlashAttention-2 backend)

**Pass Requirement**: 100% of tests must pass

### Example Usage

```bash
# Test custom kernel
python cudadent42/bench/correctness_fuzz.py

# Expected output:
# [27/27] Testing B=48, H=16, S=640, D=64... ✅ PASS
# ✅ All tests passed!
```

---

## Performance Benchmarking

**Script**: `cudadent42/bench/baseline_comprehensive.py`

**Test Configuration** (S=512 target):
- B=32, H=8, S=512, D=64
- N=100 samples
- 95% bootstrap confidence intervals
- Warmup: 20 iterations

**Metrics**:
- Median latency (ms)
- P50, P95, P99 (tail latencies)
- Coefficient of variation (CV)
- Throughput (GFLOPS)
- Memory bandwidth (GB/s)

**Warnings**:
- ⚠️  CV > 12% (high variance)
- ⚠️  GPU temp > 80°C (thermal throttling risk)

### Example Usage

```bash
# Run baseline for S=512 only
python cudadent42/bench/baseline_comprehensive.py --only s512

# Output: artifacts/baseline_comprehensive/summary.json
```

---

## CI Comparison

**Script**: `cudadent42/bench/ci_compare.py`

**Comparison Metrics**:
1. **Delta**: (candidate - baseline) / baseline × 100%
2. **CI Overlap**: Do 95% CIs overlap?
3. **Cliff's Delta**: Effect size (non-parametric)
4. **Mann-Whitney U**: Statistical significance (p < 0.05)

**Thresholds**:
- **Regression**: Δ < -3% AND CIs non-overlapping
- **Improvement**: Δ > +10% AND CIs non-overlapping AND |δ| ≥ 0.3

### Example Usage

```bash
# Compare candidate to baseline
python cudadent42/bench/ci_compare.py \
  --baseline .ci/baseline_s512.json \
  --candidate artifacts/summary.json

# Output:
# ✅ IMPROVEMENT: 12.5% faster (significant, medium effect)
# or
# ❌ REGRESSION: 4.2% slower (significant, small effect)
# or
# ⚠️  NO SIGNIFICANT DIFFERENCE: CIs overlap (Δ=+2.1%)
```

---

## Profiling

**Script**: `scripts/profile_sdpa.sh`

**Tool**: Nsight Compute (ncu)

**Metrics Collected**:
- SM throughput (% of peak)
- Tensor core activity (% of cycles)
- DRAM throughput (GB/s)
- L2 cache hit rate (%)
- Warp stall reasons
- Memory access patterns

**Output**:
- Binary report: `artifacts/ncu/sdpa_s512_b32_h8_d64.ncu-rep`
- Text summary: `artifacts/ncu/sdpa_s512_b32_h8_d64.txt`

### Example Usage

```bash
# Profile PyTorch SDPA at S=512
S=512 B=32 H=8 D=64 bash scripts/profile_sdpa.sh

# View in UI
ncu-ui artifacts/ncu/sdpa_s512_b32_h8_d64.ncu-rep
```

---

## CI Workflow

**File**: `.github/workflows/perf.yml`

**Triggers**:
- PRs touching `csrc/**`, `bench/**`, `cudadent42/**`
- Manual dispatch (`workflow_dispatch`)

**Steps**:
1. Setup environment (Ninja, ccache, CUDA)
2. **Correctness fuzzing** (must pass)
3. Benchmark S=512 config
4. Compare to baseline (`.ci/baseline_s512.json`)
5. Upload artifacts (summary JSON, Nsight reports)
6. Comment on PR with results

**Gates**:
- ❌ Fail on regression > 3%
- ❌ Fail on correctness test failure
- ⚠️  Warn on claimed improvement without statistical significance

---

## Statistical Rigor

### Confidence Intervals

**Method**: Bootstrap percentile (10,000 resamples)  
**Statistic**: Median (robust to outliers)  
**Confidence**: 95% (α = 0.05)

**Non-overlapping CIs**: Strong evidence of difference
```
Baseline:  [0.319, 0.338] ms
Candidate: [0.285, 0.295] ms  ← Non-overlapping (faster)
```

**Overlapping CIs**: Insufficient evidence of difference
```
Baseline:  [0.319, 0.338] ms
Candidate: [0.310, 0.330] ms  ← Overlapping (inconclusive)
```

### Effect Size (Cliff's Delta)

**Formula**:
$$
\delta = \frac{|\{(x_i, y_j) : x_i > y_j\}| - |\{(x_i, y_j) : x_i < y_j\}|}{n_x \cdot n_y}
$$

**Interpretation**:
- |δ| < 0.147: **negligible** (noise)
- 0.147 ≤ |δ| < 0.33: **small** (detectable but minor)
- 0.33 ≤ |δ| < 0.474: **medium** (meaningful improvement)
- |δ| ≥ 0.474: **large** (substantial improvement)

**Requirement**: For claimed improvements, |δ| ≥ 0.3 (medium or large)

### Significance Testing

**Test**: Mann-Whitney U (non-parametric, robust to skew)  
**Threshold**: p < 0.05  
**Note**: p-value alone is insufficient; must also check CIs and effect size

---

## Performance Targets

### S=512 Baseline (B=32, H=8, D=64)

**PyTorch SDPA** (FlashAttention-2):
- Median: 0.321 ms [0.3195, 0.3379]
- Throughput: 53,516 GFLOPS
- Bandwidth: 209 GB/s

**Custom Kernel Requirements**:
- **Minimum**: < 0.32 ms (no regression)
- **Improvement Claim**: < 0.29 ms (≥ 10% faster)
- **Stretch Goal**: < 0.22 ms (≥ 45% faster, 1.45× speedup)

---

## Troubleshooting

### Issue: CI fails with "Regression detected"

**Check**:
1. Is the regression real or measurement noise?
   - View CIs: If overlapping, it's noise (rerun)
   - View Cliff's δ: If |δ| < 0.147, it's negligible
2. Is GPU state stable?
   - Check warnings in artifacts
   - Look for high temp (>80°C) or high CV (>12%)

**Fix**:
- Lock GPU clocks (requires sudo)
- Increase sample size (N=100 → N=200)
- Run on clean GPU (no other processes)

### Issue: Correctness fuzzing fails

**Check**:
1. Which shapes fail? (jittered vs standard)
2. What's the error magnitude? (atol, rtol)
3. Is it FP16 precision issue? (retest with FP32)

**Fix**:
- Review kernel implementation (shared memory indexing, bounds)
- Compare intermediate values (Q@K, softmax, attention@V)
- Use `compute-sanitizer` to find memory errors

### Issue: CIs overlap but I see improvement

**Reality Check**:
- Overlapping CIs = insufficient evidence
- Need more samples (N=100 → N=200) or larger effect

**Options**:
- Increase sample size
- Test on different shape (might show clearer difference)
- Document as "potential improvement, needs validation"

---

## Best Practices

### Before Benchmarking

✅ Lock GPU clocks (if sudo available)  
✅ No other processes using GPU  
✅ Environment locked (FP16, TF32 off, deterministic)  
✅ Warmup sufficient (20+ iterations)

### During Development

✅ Profile first, optimize second  
✅ One optimization at a time (ablation study)  
✅ Test correctness after each change  
✅ Document hypothesis and result

### In PR

✅ Include Nsight evidence (`.ncu-rep` or screenshot)  
✅ Show CI comparison (include JSON)  
✅ Explain what bottleneck was addressed  
✅ Show before/after metrics

---

## References

- **Bootstrap CI**: Efron & Tibshirani (1993), "An Introduction to the Bootstrap"
- **Cliff's Delta**: Cliff (1993), "Dominance statistics: Ordinal analyses to answer ordinal questions"
- **Mann-Whitney U**: Mann & Whitney (1947), "On a test of whether one of two random variables is stochastically larger"
- **FlashAttention-2**: Dao et al. (2023), "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"

---

**Last Updated**: 2025-10-14  
**Maintainer**: GOATnote Autonomous Research Lab Initiative  
**Contact**: b@thegoatnote.com

