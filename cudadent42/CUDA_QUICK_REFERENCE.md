# CUDA Kernel Quick Reference Card

**Print this and keep it visible during GPU sessions**  
**Last Updated**: October 13, 2025

---

## 🚨 Critical Rules

| Rule | Why |
|------|-----|
| **Profile before optimize** | 80% of optimizations target wrong bottleneck |
| **Test S=32 first** | If slow on tiny config, won't be fast on large |
| **Speedup < 0.5× → STOP** | Profile with Nsight, don't guess |
| **One variable at a time** | Can't isolate impact if changing multiple things |
| **Validate after each change** | Fast but wrong is useless |

---

## ⚡ Decision Gates (Stop if Fail)

```
Gate 1: Can you import the extension?
        python3 -c "import flashmoe_science._C"
        ❌ NO → Fix build system first

Gate 2: Does shared memory fit?
        (tiles × 3 + scores) ≤ 48KB (L4) or 228KB (H100)
        ❌ NO → Reduce tile size

Gate 3: Is kernel correct? (max_diff < 0.01 vs PyTorch)
        ❌ NO → Fix correctness before performance

Gate 4: Is speedup ≥ 0.5× on S=128?
        ❌ NO → STOP → Profile with Nsight Compute
```

---

## 🔍 Profiling Commands

### Measure Single Config
```python
import torch, torch.nn.functional as F
Q = K = V = torch.randn(1,1,128,64, dtype=torch.float16, device='cuda')

# Warmup + measure
for _ in range(10): _ = F.scaled_dot_product_attention(Q,K,V)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100): O = F.scaled_dot_product_attention(Q,K,V)
end.record()
torch.cuda.synchronize()
print(f'{start.elapsed_time(end) / 100:.3f} ms')
```

### Profile with Nsight Compute
```bash
ncu --set full --launch-skip 10 --launch-count 1 \
    -o profile python3 run_kernel.py

# View report
ncu-ui profile.ncu-rep
```

---

## 📊 Nsight Compute Metrics

| Metric | Target | Fix If Low |
|--------|--------|------------|
| **SOL Memory** | >70% | Vectorize loads (float4) |
| **SOL SM** | >50% | Reduce registers/shared mem |
| **Occupancy** | >50% | Use `#pragma unroll 4` |
| **Warp Efficiency** | >90% | Reduce branch divergence |
| **Bank Conflicts** | 0 | Pad shared memory arrays |

---

## 🛠️ Common Fixes

### Memory Bandwidth < 70%
```cuda
// Before (scalar loads)
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    smem[i] = gmem[i];
}

// After (vectorized, 4× faster)
float4* gmem4 = (float4*)gmem;
float4* smem4 = (float4*)smem;
for (int i = threadIdx.x; i < N/4; i += blockDim.x) {
    smem4[i] = gmem4[i];
}
```

### Occupancy < 50%
```cuda
// Before (high register usage)
float acc[64];  // 64 registers per thread

// After (use shared memory)
extern __shared__ float smem_acc[];
```

### Launch Overhead High (kernel < 10 μs)
```cuda
// Before: Many small launches
for (tile = 0; tile < N; tile++) {
    kernel<<<1, 256>>>(tile);  // Bad
}

// After: One large launch
dim3 grid(N);
kernel<<<grid, 256>>>();  // Good
```

---

## 📐 Shared Memory Calculator

```python
def calc_smem(tile_m, tile_n, tile_k):
    tiles = 3 * tile_m * tile_k * 2  # Q,K,V in FP16
    scores = tile_m * tile_n * 4     # S in FP32
    stats = 2 * tile_m * 4           # max, sum per row
    return (tiles + scores + stats) / 1024  # KB

# Examples:
# 64×64×64:  40 KB (fits L4)
# 96×96×64:  72 KB (L4 with dynamic)
# 128×128×64: 160 KB (L4 overflow, H100 OK)
```

---

## 🎯 Performance Targets

| GPU | Bandwidth | Shared Mem | Target Speedup (S=128) |
|-----|-----------|------------|------------------------|
| L4  | 300 GB/s  | 48 KB      | 1.2× (realistic)       |
| A100| 1.5 TB/s  | 164 KB     | 1.5× (good)            |
| H100| 2.0 TB/s  | 228 KB     | 2.0× (excellent)       |

---

## 🔄 Optimization Loop (Max 3 Iterations)

```
1. PROFILE → Identify bottleneck (Nsight Compute)
   ↓
2. FIX → Apply highest-impact fix (memory, occupancy, launch)
   ↓
3. MEASURE → Re-run benchmark
   ↓
   Improved ≥ 20%? YES → Repeat (max 3×)
                   NO  → Try different fix
```

---

## ⚠️ Red Flags

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| Speedup < 0.1× | Fundamental kernel bug | Profile, don't optimize |
| Speedup decreases with S | Launch overhead | Increase tile size |
| Speedup flat across S | Already optimal | Test multi-head |
| NaN/Inf in output | Numerical instability | Use FP32 for softmax |
| Build errors | Missing template instantiation | Add explicit `template void` |

---

## 📝 Pre-Flight Checklist (5 min)

```bash
# 1. GPU available?
nvidia-smi

# 2. Extension loads?
python3 -c "import flashmoe_science._C as fa; print(dir(fa))"

# 3. Shared memory OK?
python3 shared_memory_calc.py

# 4. Correctness check (S=32)?
python3 smoke_test.py

# 5. Baseline measurement?
python3 measure_pytorch_baseline.py

# If ALL pass → Ready to benchmark
# If ANY fail → Fix before continuing
```

---

## 💰 Cost Tracking

| GPU | Cost/hour | Break-even vs Stop/Start |
|-----|-----------|--------------------------|
| L4  | $0.60     | Keep running < 5 hours   |
| A100| $1.10     | Keep running < 3 hours   |
| H100| $3.67     | Keep running < 1 hour    |

**Rule**: Stopping/starting costs $0.40-0.60 in AI context loss.

---

## 📞 Emergency Contacts

**If stuck for > 30 minutes**:
1. Read `CUDA_EXPERT_SYSTEMATIC_APPROACH.md`
2. Check `CUDA_KERNEL_LEARNING_LOOP.md`
3. Profile with Nsight Compute (don't guess!)
4. Compare to flash-attn source code

**Never**: Optimize blindly without profiling.

---

## 🎓 Learning Resources

- **Nsight Compute Docs**: https://docs.nvidia.com/nsight-compute/
- **Flash Attention 2**: https://github.com/Dao-AILab/flash-attention
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

**Print Date**: October 13, 2025  
**Version**: 1.0  
**Keep Updated**: Add new patterns as they emerge

