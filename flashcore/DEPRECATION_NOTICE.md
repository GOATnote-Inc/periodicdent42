# FlashCore Deprecation Notice

**Status**: ⚠️ **DEPRECATED** as of 2025-10-30  
**Replacement**: [BlackwellSparseK](../BlackwellSparseK/)

---

## Summary

FlashCore has been superseded by **BlackwellSparseK**, a next-generation kernel library offering superior performance, broader architecture support, and production-ready framework integrations.

All new development should use BlackwellSparseK. FlashCore will enter maintenance mode on 2025-11-15 and be fully deprecated by 2025-12-01.

---

## Why BlackwellSparseK?

| Feature | FlashCore | BlackwellSparseK |
|---------|-----------|------------------|
| **Performance** | Custom kernels, ~40-50 μs | CUTLASS 4.3.0, **<5 μs target** |
| **Architecture** | L4 (sm_89) only | **H100 (sm_90a) + Blackwell (sm_100)** |
| **Framework Integration** | None | **xFormers, vLLM V1** |
| **Deployment** | Manual | **Docker containers** |
| **Maintenance** | Custom codebase | **CUTLASS upstream support** |
| **Speedup vs SDPA** | ~1.5-2× | **5× target** |

---

## Migration Timeline

### Phase 1: Deprecation Announcement (2025-10-30)
- ✅ BlackwellSparseK v0.1.0 released
- ✅ Deprecation notice published
- ✅ Migration guide available

### Phase 2: Maintenance Mode (2025-11-15)
- FlashCore enters maintenance mode
- Bug fixes only, no new features
- All new projects must use BlackwellSparseK

### Phase 3: Full Deprecation (2025-12-01)
- FlashCore receives no updates
- Repository archived or moved to legacy/
- All documentation points to BlackwellSparseK

---

## Migration Guide

### Quick Migration

**Before (FlashCore):**
```python
from flashcore import build_baseline

# Build kernel
kernel = build_baseline(verbose=True)

# Run attention
output = kernel.forward(Q, K, V)
```

**After (BlackwellSparseK):**
```python
from blackwell_sparsek import attention_forward

# Direct API - no build step needed
output = attention_forward(Q, K, V)
```

### API Comparison

| FlashCore | BlackwellSparseK | Notes |
|-----------|------------------|-------|
| `build_baseline()` | Import only | Auto-built on first use |
| `kernel.forward(Q,K,V)` | `attention_forward(Q,K,V)` | Same semantics |
| Manual CUDA arch setup | Auto-detection | Runtime dispatch |
| `test_baseline.py` | `pytest tests/` | Standard test framework |

### Container Migration

**FlashCore** (manual setup):
```bash
cd flashcore
source scripts/env_cuda_l4.sh
python build.py
python test_baseline.py
```

**BlackwellSparseK** (containerized):
```bash
cd BlackwellSparseK
docker-compose up dev
# or
bash scripts/quick_start.sh
```

### Performance Migration

FlashCore performance baseline (L4):
- Config: B=1, H=8, S=512, D=64
- FlashCore: ~40-50 μs
- PyTorch SDPA: ~44 μs
- Speedup: ~1.1-1.5×

BlackwellSparseK performance target (H100):
- Config: B=1, H=8, S=512, D=64
- BlackwellSparseK: **<5 μs target**
- PyTorch SDPA: ~25 μs (H100 baseline)
- Speedup: **5× target**

---

## When to Use Which

### Use BlackwellSparseK If:
- ✅ Deploying on H100 or Blackwell B200
- ✅ Need <5 μs latency
- ✅ Integrating with xFormers or vLLM
- ✅ Want container-based deployment
- ✅ Starting a new project

### Continue Using FlashCore If:
- ⚠️ Locked to L4 (sm_89) hardware
- ⚠️ Legacy codebase with deep FlashCore integration
- ⚠️ Critical system that cannot be updated before 2025-12-01
- ⚠️ **Accepting performance limitations**

**Note**: Even for L4, consider BlackwellSparseK's PyTorch SDPA fallback for maintainability.

---

## Support

### FlashCore Support (Until 2025-12-01)
- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/periodicdent42/issues) (tag: `flashcore-legacy`)
- **Security Issues**: Supported until 2025-12-01
- **Feature Requests**: Not accepted (use BlackwellSparseK)

### BlackwellSparseK Support
- **Documentation**: [BlackwellSparseK/README.md](../BlackwellSparseK/README.md)
- **Migration Help**: [docs/MIGRATION_FROM_FLASHCORE.md](../BlackwellSparseK/docs/MIGRATION_FROM_FLASHCORE.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/periodicdent42/issues) (tag: `blackwellsparsek`)

---

## Resources

- **BlackwellSparseK Documentation**: [../BlackwellSparseK/](../BlackwellSparseK/)
- **Migration Guide**: [../BlackwellSparseK/docs/MIGRATION_FROM_FLASHCORE.md](../BlackwellSparseK/docs/MIGRATION_FROM_FLASHCORE.md)
- **Performance Comparison**: [../BlackwellSparseK/benchmarks/](../BlackwellSparseK/benchmarks/)
- **Quick Start**: [../BlackwellSparseK/docs/QUICKSTART.md](../BlackwellSparseK/docs/QUICKSTART.md)

---

## Acknowledgments

FlashCore served as a critical learning platform for custom CUDA kernel development in the periodicdent42 project. The lessons learned from FlashCore directly informed the design of BlackwellSparseK:

- ✅ Importance of FP16 tolerances in validation
- ✅ Benefits of tensor core utilization
- ✅ Value of systematic benchmarking
- ✅ Need for architecture-specific optimization

Thank you to all contributors who made FlashCore possible. Your work lives on in BlackwellSparseK.

---

**Questions?** Open an issue or discussion on [GitHub](https://github.com/yourusername/periodicdent42).

---

*Last Updated: 2025-10-30*  
*Deprecation Effective: 2025-10-30*  
*End of Support: 2025-12-01*

