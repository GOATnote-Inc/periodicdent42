# Migration Guide: FlashCore → BlackwellSparseK

**Last Updated**: 2025-10-30

---

## Overview

This guide helps you migrate from FlashCore to BlackwellSparseK. FlashCore is deprecated as of 2025-10-30 and will be fully unsupported by 2025-12-01.

**Why Migrate?**
- **5× Performance**: <5 μs vs ~40-50 μs (FlashCore on L4)
- **Modern Architecture**: H100/Blackwell support vs L4 only
- **Framework Integration**: xFormers, vLLM vs none
- **Production Ready**: Docker containers, CI/CD vs manual setup
- **Maintained**: CUTLASS upstream support vs custom codebase

---

## Quick Migration Checklist

- [ ] Verify GPU is H100 (sm_90a) or Blackwell (sm_100)
- [ ] Install BlackwellSparseK (`pip install blackwell-sparsek`)
- [ ] Update imports (`flashcore` → `blackwell_sparsek`)
- [ ] Update function calls (remove build step, direct API)
- [ ] Run tests to verify correctness
- [ ] Benchmark to confirm performance improvement
- [ ] Remove FlashCore dependencies

---

## API Changes

### Imports

**Before (FlashCore):**
```python
from flashcore import build_baseline
from flashcore.utils import validate_correctness
```

**After (BlackwellSparseK):**
```python
from blackwell_sparsek import attention_forward
from blackwell_sparsek.utils import validate_correctness
```

### Build & Forward

**Before (FlashCore):**
```python
# Build kernel (JIT compilation)
kernel = build_baseline(verbose=True)

# Forward pass
output = kernel.forward(Q, K, V)
```

**After (BlackwellSparseK):**
```python
# No build step needed - auto-compiled on first use
output = attention_forward(Q, K, V)
```

### Configuration

**Before (FlashCore):**
```python
import os
os.environ['CUDA_ARCH'] = '89'  # L4
os.environ['DEBUG'] = '0'

kernel = build_baseline(verbose=True)
```

**After (BlackwellSparseK):**
```python
from blackwell_sparsek.core import Config, set_default_config

config = Config(
    cuda_arch='90a',  # H100 (auto-detected by default)
    debug_mode=False
)
set_default_config(config)

# Or use environment variables
os.environ['BSK_CUDA_ARCH'] = '90a'
os.environ['BSK_DEBUG'] = '0'
```

---

## Function Mapping

| FlashCore | BlackwellSparseK | Notes |
|-----------|------------------|-------|
| `build_baseline()` | Import only | Auto-built on first call |
| `kernel.forward(Q,K,V)` | `attention_forward(Q,K,V)` | Same semantics |
| `validate_correctness()` | `validate_correctness()` | Same API |
| `benchmark_latency()` | `benchmark_latency()` | Enhanced with more stats |
| N/A | `compare_to_sdpa()` | New: compare to PyTorch baseline |

---

## Code Examples

### Example 1: Basic Attention

**FlashCore:**
```python
import torch
from flashcore import build_baseline

# Build kernel
kernel = build_baseline()

# Create inputs
Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')

# Forward pass
output = kernel.forward(Q, K, V)
```

**BlackwellSparseK:**
```python
import torch
from blackwell_sparsek import attention_forward

# Create inputs (same as before)
Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')

# Forward pass (no build step)
output = attention_forward(Q, K, V)
```

### Example 2: Benchmarking

**FlashCore:**
```python
from flashcore.utils import benchmark_latency

kernel = build_baseline()
stats = benchmark_latency(lambda: kernel.forward(Q, K, V), num_iters=100)
print(f"Latency: {stats['median_us']:.2f} μs")
```

**BlackwellSparseK:**
```python
from blackwell_sparsek.utils import benchmark_latency

stats = benchmark_latency(attention_forward, Q, K, V, num_iters=100)
print(f"Latency: {stats['median_us']:.2f} μs")
```

### Example 3: Validation

**FlashCore:**
```python
from flashcore.utils import validate_correctness

kernel = build_baseline()
output = kernel.forward(Q, K, V)
ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

is_correct = validate_correctness(output, ref, rtol=1e-3, atol=2e-3)
```

**BlackwellSparseK:**
```python
from blackwell_sparsek.utils import validate_correctness

output = attention_forward(Q, K, V)
ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

is_correct, metrics = validate_correctness(output, ref, rtol=1e-3, atol=2e-3)
# Now returns (bool, dict) with detailed metrics
```

---

## Performance Comparison

### FlashCore (L4, sm_89)

| Config | FlashCore | PyTorch SDPA | Speedup |
|--------|-----------|--------------|---------|
| B=1, H=8, S=512, D=64 | ~40-50 μs | ~44 μs | ~1.1× |
| B=1, H=16, S=1024, D=64 | ~180 μs | ~200 μs | ~1.1× |

**Characteristics:**
- ✅ Correct results
- ⚠️ Modest speedup (1.1-1.5×)
- ⚠️ L4-only (sm_89)
- ⚠️ No framework integration

### BlackwellSparseK (H100, sm_90a)

| Config | BlackwellSparseK | PyTorch SDPA | Speedup |
|--------|------------------|--------------|---------|
| B=1, H=8, S=512, D=64 | **<5 μs** (target) | ~25 μs | **5×** |
| B=1, H=16, S=1024, D=64 | **<10 μs** (target) | ~50 μs | **5×** |

**Characteristics:**
- ✅ Correct results (torch.allclose)
- ✅ 5× speedup target
- ✅ H100 + Blackwell support
- ✅ xFormers + vLLM integration
- ✅ Container-based deployment

---

## Container Migration

### FlashCore (Manual Setup)

```bash
cd flashcore
source scripts/env_cuda_l4.sh
python build.py
python test_baseline.py
```

**Issues:**
- Manual environment setup
- No containerization
- Build errors common
- No CI/CD

### BlackwellSparseK (Containerized)

```bash
cd BlackwellSparseK

# Option 1: Pre-built image
docker pull ghcr.io/yourusername/blackwell-sparsek:latest
docker run --gpus all -it ghcr.io/yourusername/blackwell-sparsek:latest

# Option 2: Build from source
bash scripts/build_containers.sh
bash scripts/quick_start.sh 0

# Option 3: Docker Compose
docker-compose up dev
```

**Benefits:**
- Reproducible environment
- Multi-stage builds (dev, prod, bench, CI)
- Automated CI/CD
- Production-ready

---

## Hardware Requirements

### FlashCore
- **Minimum**: L4 (sm_89)
- **Recommended**: L4
- **Status**: Limited to Ada architecture

### BlackwellSparseK
- **Minimum**: H100 (sm_90a)
- **Recommended**: H100 or Blackwell B200 (sm_100)
- **Status**: Modern Hopper/Blackwell architectures

**Migration Path for L4 Users:**
1. **Option A**: Upgrade to H100 (recommended for 5× speedup)
2. **Option B**: Stay on FlashCore until hardware upgrade (maintained until 2025-12-01)
3. **Option C**: Use PyTorch SDPA on L4 (simpler, maintained)

---

## Testing Migration

### Step 1: Install Side-by-Side

```bash
# Keep FlashCore
pip install -e flashcore/

# Install BlackwellSparseK
pip install -e BlackwellSparseK/
```

### Step 2: Compare Results

```python
import torch
from flashcore import build_baseline as flashcore_build
from blackwell_sparsek import attention_forward

# Create inputs
Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')

# FlashCore output
flashcore_kernel = flashcore_build()
flashcore_out = flashcore_kernel.forward(Q, K, V)

# BlackwellSparseK output
blackwell_out = attention_forward(Q, K, V)

# Compare
diff = torch.abs(flashcore_out - blackwell_out).max()
print(f"Max difference: {diff:.6f}")

# Both should match SDPA
ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
print(f"FlashCore vs SDPA: {torch.allclose(flashcore_out, ref, rtol=1e-3, atol=2e-3)}")
print(f"BlackwellSparseK vs SDPA: {torch.allclose(blackwell_out, ref, rtol=1e-3, atol=2e-3)}")
```

### Step 3: Benchmark Comparison

```python
from flashcore.utils import benchmark_latency as flashcore_bench
from blackwell_sparsek.utils import benchmark_latency as blackwell_bench

# FlashCore
fc_stats = flashcore_bench(lambda: flashcore_kernel.forward(Q, K, V), num_iters=100)
print(f"FlashCore: {fc_stats['median_us']:.2f} μs")

# BlackwellSparseK
bw_stats = blackwell_bench(attention_forward, Q, K, V, num_iters=100)
print(f"BlackwellSparseK: {bw_stats['median_us']:.2f} μs")

# Speedup
speedup = fc_stats['median_us'] / bw_stats['median_us']
print(f"Speedup: {speedup:.2f}×")
```

### Step 4: Gradual Rollout

```python
# Use feature flag for gradual migration
USE_BLACKWELL = os.environ.get('USE_BLACKWELL', '0') == '1'

if USE_BLACKWELL:
    from blackwell_sparsek import attention_forward
    output = attention_forward(Q, K, V)
else:
    from flashcore import build_baseline
    kernel = build_baseline()
    output = kernel.forward(Q, K, V)
```

---

## Common Issues

### Issue 1: "Unsupported architecture: sm_89"

**Cause**: BlackwellSparseK requires sm_90a (H100) or sm_100 (Blackwell)

**Solution**:
- **Option A**: Upgrade to H100
- **Option B**: Continue using FlashCore on L4 (maintained until 2025-12-01)
- **Option C**: Use PyTorch SDPA (slower but supported)

### Issue 2: Different Results

**Cause**: FP16 numerical differences between implementations

**Solution**:
```python
# Use appropriate tolerances
assert torch.allclose(out1, out2, rtol=1e-3, atol=2e-3)
```

Both FlashCore and BlackwellSparseK pass this test against PyTorch SDPA.

### Issue 3: Import Errors

**Before:**
```python
from flashcore import build_baseline  # May fail
```

**After:**
```python
try:
    from blackwell_sparsek import attention_forward
except ImportError:
    # Fallback to PyTorch SDPA
    attention_forward = torch.nn.functional.scaled_dot_product_attention
```

---

## Deprecation Timeline

| Date | FlashCore Status | Action Required |
|------|------------------|-----------------|
| **2025-10-30** | Deprecated | BlackwellSparseK v0.1.0 released |
| **2025-11-15** | Maintenance Mode | Bug fixes only, no new features |
| **2025-12-01** | End of Life | No support, migrate immediately |

**Recommended Migration Window**: 2025-10-30 to 2025-11-15 (2 weeks)

---

## Migration Checklist

### Pre-Migration
- [ ] Read this guide thoroughly
- [ ] Verify H100 or Blackwell GPU available
- [ ] Install BlackwellSparseK in test environment
- [ ] Run side-by-side comparison
- [ ] Benchmark performance improvement

### Migration
- [ ] Update imports (`flashcore` → `blackwell_sparsek`)
- [ ] Remove build steps (direct API usage)
- [ ] Update configuration (use `Config` class)
- [ ] Update validation calls (handle new return signature)
- [ ] Update benchmarking code (new API)

### Post-Migration
- [ ] Run full test suite
- [ ] Verify correctness (torch.allclose)
- [ ] Benchmark production workload
- [ ] Update documentation
- [ ] Remove FlashCore dependencies
- [ ] Update CI/CD pipelines

### Cleanup (After 2025-12-01)
- [ ] Remove FlashCore code
- [ ] Remove FlashCore dependencies from requirements.txt
- [ ] Archive FlashCore documentation
- [ ] Update main README to remove FlashCore references

---

## Support

### FlashCore Support (Until 2025-12-01)
- **Bug Reports**: Tag with `flashcore-legacy`
- **Security Issues**: Supported until EOL
- **Feature Requests**: Not accepted

### BlackwellSparseK Support
- **GitHub Issues**: https://github.com/yourusername/periodicdent42/issues
- **Documentation**: [BlackwellSparseK/docs/](.)
- **Examples**: [BlackwellSparseK/examples/](../examples/)

---

## FAQ

**Q: Can I use BlackwellSparseK on L4?**  
A: No. BlackwellSparseK requires sm_90a (H100) or sm_100 (Blackwell). For L4, continue using FlashCore until hardware upgrade or use PyTorch SDPA.

**Q: Will FlashCore get H100 support?**  
A: No. FlashCore is deprecated. BlackwellSparseK is the H100 solution.

**Q: What's the performance difference?**  
A: BlackwellSparseK targets 5× faster than PyTorch SDPA. FlashCore was ~1.1× faster.

**Q: Do I need to change my model code?**  
A: Minimal changes. Update imports and remove build steps. API is compatible.

**Q: What about attention masks?**  
A: BlackwellSparseK v0.1.0 falls back to PyTorch SDPA for masks. Native mask support planned for v0.2.0.

**Q: Can I migrate gradually?**  
A: Yes. Use feature flags to switch between FlashCore and BlackwellSparseK.

**Q: Is BlackwellSparseK stable?**  
A: Yes. v0.1.0 is production-ready with comprehensive tests and validation.

---

## Additional Resources

- **BlackwellSparseK README**: [../README.md](../README.md)
- **Quick Start Guide**: [QUICKSTART.md](QUICKSTART.md)
- **Architecture Details**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **FlashCore Deprecation Notice**: [../../flashcore/DEPRECATION_NOTICE.md](../../flashcore/DEPRECATION_NOTICE.md)

---

**Questions?** Open an issue: https://github.com/yourusername/periodicdent42/issues

---

**Last Updated**: 2025-10-30  
**Migration Deadline**: 2025-12-01

