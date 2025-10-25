# FlashMoE-Science Development Guide

**Complete guide to building production-grade CUDA kernels for AI-driven scientific discovery**

---

## 📋 Table of Contents

1. [Project Status](#project-status)
2. [Development Roadmap](#development-roadmap)
3. [Phase 1: Core Kernels](#phase-1-core-kernels)
4. [Phase 2: Optimization](#phase-2-optimization)
5. [Phase 3: Integration](#phase-3-integration)
6. [Phase 4: Validation](#phase-4-validation)
7. [Performance Targets](#performance-targets)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Status

### ✅ Completed (Foundation)

**Infrastructure**:
- [x] Project structure with proper organization
- [x] Build system (`setup.py` with CUDA compilation)
- [x] Python API layer (`flashmoe_science` package)
- [x] PyTorch C++ bindings (`bindings.cpp`)
- [x] Test infrastructure (pytest with CUDA support)
- [x] CI/CD pipeline (GitHub Actions)
- [x] Documentation templates

**Core Files Created**:
- [x] `flash_attention_science.cu` (stub with structure)
- [x] `fused_moe.cu` (stub)
- [x] Header files with API documentation
- [x] Python wrappers and layers

### 🚧 In Progress (Implementation)

**Phase 1: Core Kernels** (Current Focus)
- [ ] FlashAttention forward pass implementation
  - [ ] Warp specialization architecture
  - [ ] Online softmax algorithm
  - [ ] Async memory pipeline
- [ ] Fused MoE dispatch kernel
  - [ ] Radix sort implementation
  - [ ] Expert-parallel GEMM
  - [ ] Weighted combine

**Phase 2: Optimization** (Next)
- [ ] FP8 mixed precision support
- [ ] Periodic pattern-aware tiling
- [ ] Memory bandwidth optimization
- [ ] Nsight profiling and tuning

**Phase 3: Integration** (Week 3)
- [ ] vLLM backend
- [ ] SGLang kernel
- [ ] TorchTitan layers
- [ ] Megatron-LM modules

**Phase 4: Validation** (Week 4)
- [ ] Scientific benchmarks
- [ ] Performance comparisons
- [ ] Technical blog posts
- [ ] Documentation

---

## 🗺️ Development Roadmap

### Week 1-2: Core Kernels + Optimization

**Goals**:
1. Complete FlashAttention forward pass
2. Achieve 2x speedup vs PyTorch
3. Pass all correctness tests

**Key Milestones**:
- Day 1-3: Implement basic tiling and matrix multiply
- Day 4-6: Add online softmax with numerical stability
- Day 7-9: Implement warp specialization
- Day 10-12: Add async memory pipeline
- Day 13-14: Profile and optimize

**Success Criteria**:
- [ ] All tests in `test_attention_correctness.py` pass
- [ ] Max error < 1e-2 vs PyTorch SDPA
- [ ] 2x+ speedup on 2K context (H100)
- [ ] >90% SM occupancy (Nsight Compute)

### Week 3: Framework Integration

**Goals**:
1. Integrate kernels into vLLM and TorchTitan
2. Validate end-to-end correctness
3. Measure real-world performance

**Key Milestones**:
- Day 15-17: vLLM `AttentionBackend` implementation
- Day 18-19: TorchTitan layer swapping
- Day 20-21: Integration testing

**Success Criteria**:
- [ ] vLLM serves Llama-3.1-8B with custom kernels
- [ ] TorchTitan trains small model (100M params)
- [ ] End-to-end speedup measured

### Week 4: Scientific Validation + Documentation

**Goals**:
1. Run scientific benchmarks (superconductor screening)
2. Write technical blog posts
3. Create demo video
4. Finalize documentation

**Key Milestones**:
- Day 22-24: Materials discovery benchmarks
- Day 25-26: Blog post drafting
- Day 27-28: Documentation + video

**Success Criteria**:
- [ ] Superconductor screening 2.5x faster
- [ ] 3 technical blog posts published
- [ ] 10-min demo video recorded
- [ ] Complete API documentation

---

## 🔧 Phase 1: Core Kernels Implementation

### Step 1: Implement Basic Tiling (Days 1-3)

**Objective**: Get a working (but slow) implementation first.

**File**: `python/flashmoe_science/csrc/flash_attention_science.cu`

**What to Implement**:
1. Load Q tile into shared memory
2. Loop over K, V tiles
3. Compute S = Q @ K^T (naive, without warp specialization)
4. Compute softmax(S)
5. Compute O = softmax(S) @ V
6. Store output

**Key Code Locations**:
```cuda
// Line ~120: Main kernel loop
for (int tile_n = 0; tile_n < num_tiles_n; ++tile_n) {
    // Step 1: Load K, V tiles
    load_kv_tile(smem_K, smem_V, K_base, V_base, tile_n, ...);
    __syncthreads();
    
    // Step 2: Compute Q @ K^T
    compute_qk_matmul(smem_Q, smem_K, smem_S, ...);
    
    // Step 3: Compute softmax
    compute_softmax(smem_S, ...);
    
    // Step 4: Compute attention @ V
    compute_attention_v(smem_S, smem_V, acc_o, ...);
    __syncthreads();
}
```

**How to Test**:
```bash
# Build
python setup.py build_ext --inplace

# Run basic test
pytest tests/test_attention_correctness.py::TestFlashAttentionCorrectness::test_forward_vs_pytorch[torch.bfloat16-128-64] -v
```

**Expected Result**: Test passes with correct output (may be slow).

---

### Step 2: Add Online Softmax (Days 4-6)

**Objective**: Implement numerically stable online softmax algorithm.

**Why**: Avoids materializing full attention matrix, enables O(n) memory.

**Algorithm**:
```cuda
// Initialize
float m_i = -INFINITY;  // Running max
float l_i = 0.0f;       // Running sum

// For each tile
for each KV tile {
    // Compute S_tile = Q @ K_tile^T
    compute_qk_tile();
    
    // Find max in current tile
    float m_tile = max(S_tile);
    
    // Update running max
    float m_new = max(m_i, m_tile);
    
    // Compute correction factors
    float exp_i = exp(m_i - m_new);
    float exp_tile = exp(m_tile - m_new);
    
    // Update running sum
    l_i = l_i * exp_i + sum(exp(S_tile - m_tile)) * exp_tile;
    
    // Update output with correction
    O = O * exp_i + softmax(S_tile) @ V_tile * exp_tile;
    
    // Update running max
    m_i = m_new;
}

// Final normalization
O = O / l_i;
```

**Key Function**: `online_softmax_update()` (already stubbed in flash_attention_science.cu)

**How to Test**:
```bash
# Test with large values (numerical stability)
pytest tests/test_attention_correctness.py::TestFlashAttentionCorrectness::test_numerical_stability -v
```

---

### Step 3: Implement Warp Specialization (Days 7-9)

**Objective**: Use FA4-style warp specialization for parallelism.

**Architecture**:
- **Warpgroup 0** (warps 0-3): MMA operations (matrix multiply)
- **Warpgroup 1** (warps 4-7): Softmax computation
- **Warpgroup 2** (warps 8-11): Output correction

**Key Changes**:
```cuda
// At start of kernel
const int warp_id = threadIdx.x / WARP_SIZE;
const int warpgroup_id = warp_id / NUM_WARPS_PER_WARPGROUP;

// Different work for different warpgroups
if (warpgroup_id == 0) {
    // MMA warps: Compute matmuls
    for (int tile = 0; tile < num_tiles; ++tile) {
        compute_qk_matmul();  // Q @ K^T
        // Wait for softmax warps
        __syncwarp();
        compute_attention_v(); // attention @ V
    }
} else if (warpgroup_id == 1) {
    // Softmax warps: Compute softmax
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Wait for MMA warps to finish Q@K^T
        __syncwarp();
        compute_online_softmax();
    }
} else if (warpgroup_id == 2) {
    // Correction warps: Update output
    for (int tile = 0; tile < num_tiles; ++tile) {
        apply_correction_factor();
    }
}
```

**Performance Target**: 1.5x speedup from better parallelism.

---

### Step 4: Add Async Memory Pipeline (Days 10-12)

**Objective**: Overlap memory loads with computation.

**Key Pattern**:
```cuda
#include <cuda/pipeline>

cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

for (int tile = 0; tile < num_tiles; ++tile) {
    // Prefetch next tile
    if (tile + 1 < num_tiles) {
        pipe.producer_acquire();
        cuda::memcpy_async(&smem_K_next[0][0], &K_global[...], sizeof(...), pipe);
        cuda::memcpy_async(&smem_V_next[0][0], &V_global[...], sizeof(...), pipe);
        pipe.producer_commit();
    }
    
    // Compute on current tile
    compute_attention_tile(smem_K_current, smem_V_current);
    
    // Wait for next tile
    pipe.consumer_wait();
    
    // Swap buffers
    swap(smem_K_current, smem_K_next);
    swap(smem_V_current, smem_V_next);
}
```

**Performance Target**: 1.3x speedup from hiding memory latency.

---

### Step 5: Profile and Optimize (Days 13-14)

**Tools**:
```bash
# Full profiling
ncu --set full --export profile_attention python benchmarks/attention_benchmarks.py

# Open in Nsight Compute UI
ncu-ui profile_attention.ncu-rep
```

**Key Metrics to Optimize**:
1. **SM Occupancy** (target: >90%)
   - If low: Reduce register usage, reduce shared memory per block
   - Check: "Occupancy" section in Nsight

2. **Memory Bandwidth** (target: >80% of peak)
   - If low: Increase coalescing, use vectorized loads
   - Check: "Memory Workload Analysis" section

3. **Warp Efficiency** (target: >95%)
   - If low: Reduce warp divergence, balance work across warps
   - Check: "Warp State Statistics" section

4. **Instruction Efficiency**
   - If low: Remove redundant operations, use intrinsics
   - Check: "Compute Workload Analysis" section

**Optimization Checklist**:
- [ ] Vectorized loads (float4, etc.)
- [ ] Proper alignment (`__align__(16)`)
- [ ] Minimize `__syncthreads()` calls
- [ ] Use warp shuffle instead of shared memory where possible
- [ ] Unroll inner loops (`#pragma unroll`)

---

## 🎯 Performance Targets

### FlashAttention-Science

| Context Length | Batch | Heads | Dim | PyTorch (ms) | Target (ms) | Speedup |
|----------------|-------|-------|-----|-------------|-------------|---------|
| 2K             | 4     | 8     | 64  | 3.2         | 1.6         | 2.0x    |
| 4K             | 4     | 8     | 64  | 12.8        | 5.3         | 2.4x    |
| 8K             | 4     | 8     | 64  | 51.2        | 19.2        | 2.7x    |

### Hardware Utilization

| Metric                     | Target  | How to Measure                           |
|----------------------------|---------|------------------------------------------|
| SM Occupancy               | >90%    | Nsight Compute: Occupancy section        |
| Memory Bandwidth           | >80%    | Nsight Compute: Memory Workload Analysis |
| Warp Efficiency            | >95%    | Nsight Compute: Warp State Statistics    |
| Arithmetic Intensity       | >10     | (FLOPs) / (Bytes transferred)            |

---

## 🐛 Troubleshooting

### Issue: Tests fail with "CUDA extensions not available"

**Solution**:
```bash
# Rebuild extensions
python setup.py build_ext --inplace --force

# Verify import
python -c "from flashmoe_science import _C; print('✓ Extensions loaded')"
```

### Issue: Numerical errors (large max_error)

**Cause**: Online softmax implementation bug

**Debug**:
1. Print intermediate values in kernel
2. Compare tile-by-tile with PyTorch
3. Check overflow in exp() calls

**Solution**: Ensure proper scaling in online softmax update.

### Issue: Kernel timeout or crash

**Cause**: Infinite loop or out-of-bounds access

**Debug**:
```bash
# Run with CUDA error checking
CUDA_LAUNCH_BLOCKING=1 pytest tests/test_attention_correctness.py -v

# Use cuda-memcheck
cuda-memcheck python -c "import tests.test_attention_correctness"
```

### Issue: Low performance (<1.5x speedup)

**Cause**: Not using hardware efficiently

**Debug**:
1. Profile with Nsight Compute
2. Check occupancy (should be >70%)
3. Check memory bandwidth utilization
4. Look for warp divergence

**Solution**: Follow optimization checklist in Step 5.

---

## 📚 Recommended Resources

### CUDA Programming
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [GPU MODE Lectures](https://www.youtube.com/@GPUMODE) - Excellent CUDA tutorials
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### FlashAttention Papers
- [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) (Shah et al., 2024)

### Implementation References
- [FlashAttention-2 GitHub](https://github.com/Dao-AILab/flash-attention)
- [Modal: Reverse Engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)

### Community
- [GPU MODE Discord](https://discord.gg/gpumode) - Active CUDA community
- [r/CUDA Reddit](https://reddit.com/r/CUDA)
- [PyTorch Forums](https://discuss.pytorch.org)

---

## 🎓 Next Steps

### Immediate (This Week)
1. **Implement basic tiling** (Step 1 above)
2. **Get first test passing** (even if slow)
3. **Add online softmax** (Step 2)
4. **Profile initial version** with Nsight

### Short Term (Next 2 Weeks)
1. Complete warp specialization
2. Add async memory pipeline
3. Optimize based on profiling
4. Start vLLM integration

### Long Term (Month 2+)
1. Implement backward pass (for training)
2. Add FP8 support (Hopper-specific)
3. Port to AMD ROCm (show multi-vendor expertise)
4. Publish blog posts and benchmarks

---

## 💡 Pro Tips

1. **Start simple**: Get correctness first, optimize later
2. **Test incrementally**: Run tests after every change
3. **Profile religiously**: Never guess where the bottleneck is
4. **Read existing code**: FA2 implementation is excellent reference
5. **Ask for help**: GPU MODE Discord community is very helpful
6. **Document as you go**: Explain WHY, not just WHAT

---

**Remember**: This project demonstrates your expertise, not just performance numbers. Clean code, good tests, and thorough documentation matter as much as speed.

**Good luck building FlashMoE-Science!** 🚀

