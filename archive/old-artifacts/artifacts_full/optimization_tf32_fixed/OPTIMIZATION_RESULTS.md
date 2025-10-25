# SOTA Optimization Results

**GPU**: NVIDIA L4
**Date**: 2025-10-13 23:21:57
**Target Shape**: B=32, H=8, S=512, D=64
**Iterations**: 100 (warmup: 20)

---

## 🏁 Baseline

**Backend**: auto
**Median**: 0.3226 ms
**95% CI**: [0.3205, 0.3246] ms
**Throughput**: 53261.0 GFLOPS
**Bandwidth**: 208.1 GB/s

## 🏆 Result

Baseline configuration with auto backend is optimal for PyTorch SDPA.

### 📝 Publication-Ready Statement

Using PyTorch SDPA (auto backend) on NVIDIA L4 (FP16), achieved 0.3226 ms (95% CI: [0.3205, 0.3246]) for fixed shape B=32, H=8, S=512, D=64 (N=100). Environment locked (TF32 off, deterministic algorithms on).

## 🔬 Reproducibility

- Environment locked: TF32 disabled, deterministic algorithms enabled
- Bootstrap CIs: 10,000 resamples, seed=42
- Measurements: Median of 100 iterations (20 warmup)
- Environment fingerprint: `cudadent42/bench/artifacts/optimization_tf32_fixed/env.json`
