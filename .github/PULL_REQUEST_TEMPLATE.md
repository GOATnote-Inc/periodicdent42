# Pull Request

## Description

<!-- Provide a brief description of the changes in this PR -->

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Performance optimization (non-breaking change which improves performance)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

---

## ⚡ Performance Checks

<!-- Required for changes to CUDA kernels, benchmarking, or optimization -->

- [ ] Correctness fuzzing passed (27 configs)
- [ ] Baseline benchmark run (N=100 samples)
- [ ] Statistical comparison performed (bootstrap CIs, effect sizes)
- [ ] Nsight profiling captured (if claiming >10% improvement)
- [ ] Performance regression < 3% (or justified)

---

## ⚡ Performance Intent & Hypothesis

**Target Shapes**: <!-- e.g., B=32, H=8, S=512, D=64 -->

**Bottleneck Hypothesis**: <!-- e.g., Low tensor core utilization (57% → target 80%+) -->

**Proposed Fix**: <!-- e.g., Increase block size from 128 to 256 threads -->

### Nsight Evidence (Before/After)

**Baseline** (commit: `XXXXXXX`):
- Tensor Core Utilization: XX%
- DRAM Throughput: XX% of peak
- L2 Hit Rate: XX%
- Latency: X.XXX ms [CI: X.XXX, X.XXX]

**After This PR**:
- Tensor Core Utilization: XX%
- DRAM Throughput: XX% of peak  
- L2 Hit Rate: XX%
- Latency: X.XXX ms [CI: X.XXX, X.XXX]

**Artifacts**: `bench/artifacts/ncu/before_vs_after.ncu-rep`

### Statistical Results

- **Median Δ**: +X.X% (X.XXX ms → X.XXX ms)
- **95% CI Overlap**: Yes/No
- **Cliff's δ**: X.XXX (negligible/small/medium/large)
- **Mann-Whitney U p-value**: X.XXXX
- **Verdict**: Faster/Maintained/Slower

---

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Correctness fuzzing passes (if kernel changes)
- [ ] Performance regression < 3% (if optimization)

---

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings or errors
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

---

## Additional Context

<!-- Add any other context, screenshots, or references about the PR here -->
