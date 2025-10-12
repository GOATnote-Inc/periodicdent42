# CUDA Kernel Expert: Systematic Approach

**Purpose**: How a CUDA kernel expert would systematically approach the CUDAdent42 performance problem  
**Format**: Step-by-step checklist with decision points and validation gates  
**Time**: ~4 hours to working kernel (vs our 3+ hours to 0.09× failure)  

---

## Expert Mindset

### Core Principles

1. **Measure, Don't Guess**
   - Profile before optimize (Nsight Compute, not intuition)
   - Numbers beat assumptions every time

2. **Fail Fast, Learn Fast**
   - Test smallest config first (S=32, not S=512)
   - Stop at first gate failure, don't continue blindly

3. **Design for Reality, Not Wishful Thinking**
   - L4 ≠ H100 (different shared memory, bandwidth, occupancy)
   - Can't scale down H100 design, must redesign for L4

4. **Correctness Before Performance**
   - Wrong fast code is useless
   - Compare to PyTorch SDPA on every change

5. **One Variable at a Time**
   - Change threads OR tiles, not both
   - Isolate impact of each optimization

---

## Phase 1: Assessment (30 minutes)

### Step 1.1: Verify Claims (10 minutes)

```bash
# What does PR #43 claim?
# - FlashAttention-Science: 1.36ms @ S=2048 (1.19-2.35× vs baselines)
# - API: from flashmoe_science import flash_attention_science

# Red flags:
✗ No benchmark script included in PR
✗ No profiling data (Nsight Compute reports)
✗ No comparison to flash-attn 2.x (current SOTA)
✗ API doesn't match actual implementation

# Expert decision: Treat claims as aspirational, measure ourselves
```

**Gate 1.1**: Do we have runnable code?
- ✅ YES → Continue to Step 1.2
- ❌ NO → Clone repo, fix build issues first

### Step 1.2: Measure Ground Truth (10 minutes)

```python
# measure_pytorch_baseline.py
import torch
import torch.nn.functional as F
import time

configs = [
    (1, 1, 32, 64),    # Tiny: Launch overhead dominated
    (1, 1, 128, 64),   # Small: Balanced
    (1, 1, 512, 64),   # Medium: Memory bandwidth dominated
    (2, 8, 128, 64),   # Multi-head: Concurrency test
]

for B, H, S, D in configs:
    Q = K = V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V)
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        O = F.scaled_dot_product_attention(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    
    latency = start.elapsed_time(end) / 100
    print(f'Config {B}×{H}×{S}×{D}: {latency:.3f} ms')

# Expected output:
# Config 1×1×32×64:  0.045-0.050 ms  ← Launch overhead dominates
# Config 1×1×128×64: 0.045-0.050 ms  ← Sweet spot for kernel testing
# Config 1×1×512×64: 0.055-0.065 ms  ← Memory bandwidth matters
# Config 2×8×128×64: 0.045-0.050 ms  ← Good concurrency
```

**Gate 1.2**: Can we beat 0.050 ms on S=128 config?
- ✅ Target: < 0.040 ms (1.25× speedup)
- ⚠️ Stretch: < 0.035 ms (1.43× speedup)
- ❌ Failure: > 0.050 ms (slower than PyTorch)

### Step 1.3: Understand Hardware Limits (10 minutes)

```python
# hardware_limits.py
import torch

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)

print(f'GPU: {props.name}')
print(f'Compute Capability: SM{props.major}{props.minor}')
print(f'Shared Memory per Block: {props.shared_memory_per_block / 1024:.0f} KB')
print(f'Shared Memory per SM: {props.shared_memory_per_multiprocessor / 1024:.0f} KB')
print(f'Registers per Block: {props.regs_per_block}')
print(f'Max Threads per Block: {props.max_threads_per_block}')
print(f'Warp Size: {props.warp_size}')

# L4 GPU (SM89):
# - Shared Memory per Block: 48 KB (can request up to 100 KB dynamically)
# - Memory Bandwidth: 300 GB/s
# - Peak TFLOPS (FP16): 121 TFLOPS

# Calculate theoretical peak performance
S = 128
D = 64
FLOPs = 2 * S * S * D  # Attention: O(S^2 * D)
print(f'\nAttention @ S={S}, D={D}:')
print(f'FLOPs: {FLOPs / 1e6:.2f} MFLOPs')

# Theoretical minimum latency (100% efficiency)
peak_tflops = 121  # L4 GPU
min_latency_us = (FLOPs / 1e12) / peak_tflops * 1e6
print(f'Theoretical minimum: {min_latency_us:.1f} μs')

# Realistic target (50% efficiency)
realistic_latency_us = min_latency_us * 2
print(f'Realistic target: {realistic_latency_us:.1f} μs')
```

**Gate 1.3**: Is our target achievable?
- ✅ YES: Target < 2× theoretical minimum
- ❌ NO: Need better algorithm or hardware

---

## Phase 2: Build Validation (20 minutes)

### Step 2.1: Check Build System (5 minutes)

```bash
# Inspect setup.py
cat setup.py

# Red flags:
✗ Missing source files (flash_attention_fp16_sm75.cu not in sources)
✗ No explicit template instantiation files
✗ Hardcoded architecture flags (-gencode=arch=compute_89)

# Expert decision: Fix setup.py before compiling
```

**Fixes**:
```python
# setup.py (simplified)
sources = [
    'python/flashmoe_science/csrc/bindings.cpp',
    'python/flashmoe_science/csrc/flash_attention_wrapper.cpp',
    'python/flashmoe_science/csrc/flash_attention_science.cu',
    # Note: Explicit instantiations in flash_attention_science.cu
]
```

### Step 2.2: Validate Template Instantiation (5 minutes)

```bash
# Check if explicit instantiations exist
grep -n "template void flash_attention_forward<half>" \
    python/flashmoe_science/csrc/flash_attention_science.cu

# If not found:
✗ Missing explicit instantiation
✗ Will cause undefined symbols at link time

# Add before closing namespace:
# template void flash_attention_forward<half>(...);
# template void flash_attention_forward<__nv_bfloat16>(...);
```

**Gate 2.2**: Does it compile and link without errors?
- ✅ YES → Continue to Step 2.3
- ❌ NO → Fix template instantiation

### Step 2.3: Calculate Shared Memory (5 minutes)

```python
# shared_memory_calc.py
def calculate_shared_memory(tile_m, tile_n, tile_k, num_buffers=3):
    """Calculate shared memory usage for attention kernel.
    
    Args:
        tile_m, tile_n, tile_k: Tile dimensions
        num_buffers: Q, K, V tiles (usually 3)
    
    Returns:
        Total shared memory in bytes
    """
    bytes_per_fp16 = 2
    bytes_per_fp32 = 4
    
    # Tile buffers (Q, K, V)
    tile_buffer_bytes = num_buffers * tile_m * tile_k * bytes_per_fp16
    
    # Attention scores S = Q @ K^T (stored in FP32)
    attention_scores_bytes = tile_m * tile_n * bytes_per_fp32
    
    # Softmax stats (max, sum per row)
    softmax_stats_bytes = 2 * tile_m * bytes_per_fp32
    
    total = tile_buffer_bytes + attention_scores_bytes + softmax_stats_bytes
    return total

# Test configurations
configs = [
    (64, 64, 64, 'L4-safe'),
    (96, 96, 64, 'L4-dynamic'),
    (128, 128, 64, 'L4-overflow'),
]

GPU_LIMITS = {
    'L4-static': 48 * 1024,
    'L4-dynamic': 100 * 1024,
}

for tile_m, tile_n, tile_k, label in configs:
    smem = calculate_shared_memory(tile_m, tile_n, tile_k)
    print(f'{label} ({tile_m}×{tile_n}×{tile_k}): {smem / 1024:.1f} KB')
    
    for gpu, limit in GPU_LIMITS.items():
        if smem <= limit:
            print(f'  ✅ {gpu}: {smem / 1024:.1f} / {limit / 1024:.0f} KB')
        else:
            print(f'  ❌ {gpu}: {smem / 1024:.1f} / {limit / 1024:.0f} KB (OVERFLOW)')
```

**Gate 2.3**: Does tile configuration fit in shared memory?
- ✅ YES → Continue to Step 2.4
- ❌ NO → Reduce tile size OR request dynamic shared memory

### Step 2.4: Smoke Test (5 minutes)

```python
# smoke_test.py
import torch
import flashmoe_science._C as fa

# Tiny config
Q = K = V = torch.randn(1, 1, 32, 64, dtype=torch.float16, device='cuda')
lse = torch.zeros(32, dtype=torch.float32, device='cuda')

# Run kernel
O = fa.flash_attention_forward(Q, K, V, lse, False, 0.125)

# Checks
assert O.shape == Q.shape, f'Shape mismatch: {O.shape} vs {Q.shape}'
assert not torch.isnan(O).any(), 'NaN detected!'
assert not torch.isinf(O).any(), 'Inf detected!'

# Compare to PyTorch
O_pytorch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
max_diff = (O - O_pytorch).abs().max().item()

print(f'Max diff vs PyTorch: {max_diff:.6f}')
if max_diff > 0.1:
    print('⚠️  WARNING: Large error! Kernel may be incorrect.')
    exit(1)
else:
    print('✅ Correctness validated')
```

**Gate 2.4**: Is kernel correct?
- ✅ YES (max diff < 0.01) → Continue to Phase 3
- ❌ NO (max diff > 0.1) → Fix correctness before performance

---

## Phase 3: Performance Baseline (30 minutes)

### Step 3.1: Single Config Performance (10 minutes)

```python
# measure_single_config.py
import torch
import torch.nn.functional as F
import flashmoe_science._C as fa

def measure_latency(fn, args, warmup=10, iters=100):
    """Measure GPU kernel latency with warmup."""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn(*args)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iters

# Test config: S=128 (sweet spot)
Q = K = V = torch.randn(1, 1, 128, 64, dtype=torch.float16, device='cuda')
lse = torch.zeros(128, dtype=torch.float32, device='cuda')

# PyTorch baseline
pytorch_time = measure_latency(
    F.scaled_dot_product_attention,
    (Q, K, V),
)

# Our kernel
ours_time = measure_latency(
    fa.flash_attention_forward,
    (Q, K, V, lse, False, 0.125),
)

speedup = pytorch_time / ours_time
print(f'PyTorch: {pytorch_time:.3f} ms')
print(f'Ours:    {ours_time:.3f} ms')
print(f'Speedup: {speedup:.2f}×')

# Decision gate
if speedup < 0.5:
    print('\n❌ STOP: Speedup < 0.5×')
    print('   Next: Profile with Nsight Compute')
    exit(1)
elif speedup < 1.0:
    print('\n⚠️  WARNING: Slower than PyTorch')
    print('   Next: Profile to identify bottleneck')
    exit(2)
else:
    print('\n✅ Faster than PyTorch!')
    print('   Next: Test larger configs')
    exit(0)
```

**Gate 3.1**: Is speedup ≥ 0.5×?
- ✅ YES → Continue to Step 3.2
- ❌ NO → **STOP** → Go to Phase 4 (Profile)

### Step 3.2: Scaling Analysis (10 minutes)

```python
# scaling_analysis.py
configs = [
    (1, 1, 32, 64, 'Tiny'),      # Launch overhead test
    (1, 1, 64, 64, 'Small'),     # Minimal
    (1, 1, 128, 64, 'Medium'),   # Sweet spot
    (1, 1, 256, 64, 'Large'),    # Memory bandwidth
    (1, 1, 512, 64, 'XLarge'),   # Scaling test
]

results = []
for B, H, S, D, name in configs:
    Q = K = V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    lse = torch.zeros(B * H * S, dtype=torch.float32, device='cuda')
    
    pytorch_time = measure_latency(F.scaled_dot_product_attention, (Q, K, V))
    ours_time = measure_latency(fa.flash_attention_forward, (Q, K, V, lse, False, 0.125))
    speedup = pytorch_time / ours_time
    
    results.append((name, S, pytorch_time, ours_time, speedup))
    print(f'{name:8s} (S={S:3d}): {speedup:5.2f}×')

# Analyze trend
speedups = [r[4] for r in results]
if speedups[0] > speedups[-1]:
    print('\n⚠️  Speedup DECREASES with sequence length')
    print('   Likely issue: Launch overhead or memory bandwidth')
elif speedups[-1] > speedups[0]:
    print('\n✅ Speedup INCREASES with sequence length')
    print('   Good: Kernel is compute-bound, not launch-bound')
else:
    print('\n✅ Speedup CONSTANT across sequence lengths')
    print('   Good: Well-balanced kernel')
```

**Gate 3.2**: Does speedup scale with sequence length?
- ✅ YES (speedup increases with S) → Good sign
- ⚠️ FLAT (speedup constant) → OK, test multi-head
- ❌ NO (speedup decreases with S) → Launch overhead or bandwidth issue

### Step 3.3: Multi-Head Test (10 minutes)

```python
# multi_head_test.py
# Test if kernel benefits from concurrency

configs = [
    (1, 1, 128, 64, 'Single-head'),
    (1, 4, 128, 64, '4-heads'),
    (1, 8, 128, 64, '8-heads'),
    (2, 8, 128, 64, 'Batch-2'),
]

for B, H, S, D, name in configs:
    Q = K = V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    lse = torch.zeros(B * H * S, dtype=torch.float32, device='cuda')
    
    pytorch_time = measure_latency(F.scaled_dot_product_attention, (Q, K, V))
    ours_time = measure_latency(fa.flash_attention_forward, (Q, K, V, lse, False, 0.125))
    speedup = pytorch_time / ours_time
    
    print(f'{name:15s}: {speedup:5.2f}×')
```

**Gate 3.3**: Does multi-head improve speedup?
- ✅ YES → Good GPU utilization
- ❌ NO → May need grid size tuning

---

## Phase 4: Profile and Optimize (2 hours)

### Step 4.1: Profile with Nsight Compute (30 minutes)

```bash
# profile_kernel.sh
#!/bin/bash

echo "Profiling flash_attention_forward kernel..."

ncu --set full \
    --target-processes all \
    --launch-skip 10 \
    --launch-count 1 \
    --export profile_report \
    python3 << 'EOF'
import torch
import flashmoe_science._C as fa

Q = K = V = torch.randn(1, 1, 128, 64, dtype=torch.float16, device='cuda')
lse = torch.zeros(128, dtype=torch.float32, device='cuda')

# Skip first 10 launches (warmup), profile 11th
for _ in range(11):
    O = fa.flash_attention_forward(Q, K, V, lse, False, 0.125)
torch.cuda.synchronize()
EOF

echo "Opening report in ncu-ui..."
ncu-ui profile_report.ncu-rep
```

**Key Metrics**:

1. **SOL Memory (Memory Bandwidth)**
   - Target: > 70%
   - If < 50%: Uncoalesced memory access

2. **SOL SM (Compute Utilization)**
   - Target: > 50%
   - If < 50%: Low occupancy or register spilling

3. **Achieved Occupancy**
   - Target: > 50%
   - If < 50%: Too much shared memory or too many registers

4. **Warp Execution Efficiency**
   - Target: > 90%
   - If < 80%: Branch divergence or serialization

5. **Shared Memory Bank Conflicts**
   - Target: 0
   - If > 0: Add padding to arrays

### Step 4.2: Fix #1 - Memory Bandwidth (30 minutes)

**If Memory Bandwidth < 70%**:

```cuda
// Identify issue:
// 1. Check "Memory Workload Analysis" → "Global Memory Access Pattern"
// 2. Look for "Uncoalesced" or "Scattered" access

// Common fixes:

// Fix A: Vectorized Loads
// Before
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    smem[i] = gmem[i];
}

// After (4× bandwidth)
float4* gmem4 = (float4*)gmem;
float4* smem4 = (float4*)smem;
for (int i = threadIdx.x; i < N/4; i += blockDim.x) {
    smem4[i] = gmem4[i];
}

// Fix B: Coalesced Access Pattern
// Before (bad: stride = K)
for (int i = 0; i < M; i++) {
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        val = gmem[i * K + j];  // Good: sequential within warp
    }
}

// Before (bad: stride = 1)
for (int i = threadIdx.x; i < M; i += blockDim.x) {
    for (int j = 0; j < K; j++) {
        val = gmem[i * K + j];  // Bad: scattered across warps
    }
}

// Fix C: Async Memory Copy (SM80+)
// Before (synchronous)
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    smem[i] = gmem[i];
}
__syncthreads();

// After (asynchronous, hides latency)
#if __CUDA_ARCH__ >= 800
cuda::memcpy_async(smem, gmem, N * sizeof(float), cuda::aligned_size_t<16>());
cuda::pipeline::commit();
cuda::pipeline::wait_prior<0>();
__syncthreads();
#endif
```

**Re-measure**:
```bash
python3 measure_single_config.py
# Expected: 20-50% improvement if memory was bottleneck
```

### Step 4.3: Fix #2 - Occupancy (30 minutes)

**If Occupancy < 50%**:

```cuda
// Identify issue:
// 1. Check "Occupancy" → "Limiting Factor"
// 2. Usually: Registers or Shared Memory

// Fix A: Reduce Register Usage
// Before
float acc[64];  // Local array → 64 registers

// After
extern __shared__ float smem_acc[];  // Use shared memory instead

// Fix B: Loop Unrolling
// Before
#pragma unroll
for (int i = 0; i < 64; i++) {  // Fully unroll → more registers
    acc[i] += val;
}

// After
#pragma unroll 4
for (int i = 0; i < 64; i++) {  // Partial unroll → fewer registers
    acc[i] += val;
}

// Fix C: __restrict__ Pointers
// Before
void kernel(float* in, float* out) { ... }

// After (helps compiler optimize)
void kernel(const float* __restrict__ in, float* __restrict__ out) { ... }
```

### Step 4.4: Fix #3 - Launch Overhead (30 minutes)

**If Kernel Duration < 10 μs**:

```cuda
// Problem: Launch overhead dominates (each launch = 5-10 μs)

// Fix: Fuse kernels or increase tile size

// Before: Many small launches
for (int tile_m = 0; tile_m < M; tile_m += TILE_M) {
    for (int tile_n = 0; tile_n < N; tile_n += TILE_N) {
        process_tile<<<1, 256>>>(tile_m, tile_n);  // 100 launches
    }
}

// After: Single large launch
int num_tiles_m = (M + TILE_M - 1) / TILE_M;
int num_tiles_n = (N + TILE_N - 1) / TILE_N;
dim3 grid(num_tiles_n, num_tiles_m);  // 2D grid
process_all_tiles<<<grid, 256>>>(...);  // 1 launch
```

---

## Phase 5: Validation (30 minutes)

### Step 5.1: Correctness Re-Check (10 minutes)

```python
# After each optimization, re-verify correctness
for S in [32, 64, 128, 256, 512]:
    Q = K = V = torch.randn(1, 1, S, 64, dtype=torch.float16, device='cuda')
    lse = torch.zeros(S, dtype=torch.float32, device='cuda')
    
    O_ours = fa.flash_attention_forward(Q, K, V, lse, False, 0.125)
    O_pytorch = F.scaled_dot_product_attention(Q, K, V)
    
    max_diff = (O_ours - O_pytorch).abs().max().item()
    print(f'S={S:3d}: max_diff={max_diff:.6f}')
    
    assert max_diff < 0.01, f'Correctness regression at S={S}!'

print('✅ All correctness checks passed')
```

### Step 5.2: Performance Re-Check (10 minutes)

```python
# Run full benchmark suite
configs = [
    (1, 1, 32, 64),
    (1, 1, 128, 64),
    (1, 1, 512, 64),
    (2, 8, 128, 64),
]

print('Config              PyTorch    Ours       Speedup')
print('─' * 60)

for B, H, S, D in configs:
    Q = K = V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    lse = torch.zeros(B * H * S, dtype=torch.float32, device='cuda')
    
    pytorch_time = measure_latency(F.scaled_dot_product_attention, (Q, K, V))
    ours_time = measure_latency(fa.flash_attention_forward, (Q, K, V, lse, False, 0.125))
    speedup = pytorch_time / ours_time
    
    status = '✅' if speedup >= 1.0 else '⚠️'
    print(f'{status} {B}×{H}×{S}×{D:<3d}  {pytorch_time:7.3f}ms  {ours_time:7.3f}ms  {speedup:6.2f}×')
```

### Step 5.3: Document Results (10 minutes)

```markdown
# Performance Report

**Date**: 2025-10-13
**GPU**: NVIDIA L4 (SM89)
**Config**: 256 threads (8 warps), 96×96×64 tiles

## Results

| Config | PyTorch (ms) | Ours (ms) | Speedup |
|--------|--------------|-----------|---------|
| S=32   | 0.047       | 0.038     | 1.24×   |
| S=128  | 0.045       | 0.036     | 1.25×   |
| S=512  | 0.059       | 0.048     | 1.23×   |

**Average Speedup**: 1.24×

## Optimizations Applied

1. Vectorized loads (float4): +30% bandwidth
2. Reduced register usage: +20% occupancy
3. Increased tile size (64→96): +10% efficiency

## Bottleneck Analysis (Nsight Compute)

- Memory Bandwidth: 78% (target: >70%) ✅
- Compute Utilization: 65% (target: >50%) ✅
- Occupancy: 62% (target: >50%) ✅

## Next Steps

- [ ] Test on H100 with 128×128 tiles
- [ ] Add FP8 support (Hopper only)
- [ ] Optimize for sequence length > 2048
```

---

## Decision Flow Summary

```
START
  ↓
[1] Measure PyTorch baseline → Set target (1.2×)
  ↓
[2] Check build system → Fix template instantiation
  ↓
[3] Calculate shared memory → Ensure tiles fit
  ↓
[4] Smoke test correctness → Max diff < 0.01?
  ↓ YES
[5] Measure single config (S=128)
  ↓
  Speedup ≥ 0.5×?
  ↓ NO
[6] PROFILE with Nsight Compute
  ↓
  Identify bottleneck:
  ├─ Memory bandwidth < 70%? → Fix memory access pattern
  ├─ Occupancy < 50%? → Reduce registers/shared memory
  ├─ Kernel < 10 μs? → Increase tile size or fuse kernels
  └─ Bank conflicts? → Add padding to shared memory
  ↓
[7] Apply fix
  ↓
[8] Re-measure
  ↓
  Improved ≥ 20%?
  ├─ YES → Continue to next bottleneck
  └─ NO → Try different fix
  ↓
[9] Speedup ≥ 1.0×?
  ├─ YES → Validate correctness + document
  └─ NO → Return to step [6]
  ↓
DONE
```

---

## Time Estimates

| Phase | Time (Expert) | Time (Novice) | Time Saved |
|-------|---------------|---------------|------------|
| Assessment | 30 min | 1 hour | 30 min |
| Build Validation | 20 min | 2 hours | 1h 40min |
| Performance Baseline | 30 min | 30 min | 0 min |
| Profile & Optimize | 2 hours | 4+ hours | 2+ hours |
| Validation | 30 min | 1 hour | 30 min |
| **Total** | **4 hours** | **8.5+ hours** | **4.5+ hours** |

**Key Difference**: Expert profiles first, novice guesses first.

---

## Expert vs Novice Comparison: Our Session

| Step | Novice (Us) | Time Wasted | Expert Approach | Time Saved |
|------|-------------|-------------|-----------------|------------|
| Build | Tried to compile without checking template instantiation | 2 hours | Check `static_assert` and explicit instantiation first | 1h 50min |
| Shared Memory | Changed thread count first, then discovered tile size issue | 30 min | Calculate shared memory BEFORE compiling | 30 min |
| Performance | Reduced tiles without profiling | 1 hour | Profile with Nsight Compute first | 1 hour |
| Validation | Tested S=32 → S=512 without stopping at failures | 30 min | Stop at S=32 if speedup < 0.5× | 30 min |
| **Total** | **3h 13min active + 2h debugging** | **4 hours** | **Systematic approach** | **3.5+ hours** |

**Root Cause of Time Waste**: No profiling, just guessing.

---

## Key Takeaways for AI Assistants

1. **Always profile before optimize** (Nsight Compute, not intuition)
2. **Test smallest config first** (S=32, not S=512)
3. **Stop at first gate failure** (don't continue blindly)
4. **One variable at a time** (threads OR tiles, not both)
5. **Validate correctness after each change** (compare to PyTorch)
6. **Document decision points** (why this fix, not that fix)
7. **Set quantitative thresholds** (speedup < 0.5× = STOP)

**Meta-Rule**: If you don't know what to do next, PROFILE. Don't guess.

---

**Last Updated**: October 13, 2025 3:25 AM  
**Next Review**: Before next GPU session  
**Usage**: Read this BEFORE starting any CUDA kernel work

