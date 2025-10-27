# ⚡ WGMMA Quick Reference Card
## H100 Native Implementation - Critical Facts & Patterns

---

## 🎯 WGMMA INSTRUCTION REFERENCE

### Available Shapes
```
m64n8k16, m64n16k16, m64n32k16, m64n64k16   (most common: m64n64k16)
m64n128k16, m64n256k16                       (H100 only, large N)
```

### Data Type Combinations (FP16 → FP32)
```cuda
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
// Output: FP32 accumulator (32 regs per thread)
// Input A: FP16 matrix (64×16)
// Input B: FP16 matrix (64×16 or 16×64 if transposed)
```

---

## 📐 THREAD-TO-OUTPUT MAPPING (m64n64k16)

### Critical Pattern
```cuda
// 128 threads (warp group) → 64×64 output (4096 values)
// Each thread: 32 FP32 registers (128 × 32 = 4096)

const int warp_id = tid / 32;         // 0-3 (4 warps)
const int lane = tid % 32;            // 0-31 within warp
const int warp_row_base = warp_id * 16;  // Each warp handles 16 rows

// Register pair mapping (simplified, see PTX ISA 9.7.13.7 for exact)
for (int i = 0; i < 32; i += 2) {
    int reg_pair_id = i / 2;          // 0-15
    int row = warp_row_base + (lane % 8) + ((reg_pair_id / 8) * 8);
    int col_section = (reg_pair_id % 8) * 8;
    int col = col_section + (lane / 8) * 2;
    
    C[row * 64 + col + 0] = acc[i];
    C[row * 64 + col + 1] = acc[i + 1];
}
```

**Reference:** PTX ISA Section 9.7.13.7

---

## 🧮 DESCRIPTOR ENCODING

### 64-bit Descriptor Layout
```
Bits [19:0]  : Shared memory address (128-byte aligned)
Bits [31:20] : Reserved (must be 0)
Bits [45:32] : Leading dimension in 16B units
Bits [48:46] : Swizzle mode (0=none, 1=32B, 2=64B, 3=128B)
Bits [63:49] : Reserved (must be 0)
```

### Descriptor Creation
```cuda
__device__ uint64_t make_smem_desc(
    const void* smem_ptr, 
    uint32_t leading_dim,    // In elements (not bytes)
    uint32_t swizzle = 3     // Use 3 for best performance
) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    uint32_t ld_units = (leading_dim * sizeof(__half)) / 16;
    
    return (addr & 0xFFFFF) |
           ((uint64_t)(ld_units & 0x3FFF) << 32) |
           ((uint64_t)(swizzle & 0x7) << 46);
}
```

---

## 🎛️ FENCE OPERATIONS

### Correct Ordering
```cuda
// 1. Fence BEFORE descriptor creation
wgmma_fence();

// 2. Create descriptors
uint64_t desc_a = make_smem_desc(&smem_A[0][0], 32, 3);
uint64_t desc_b = make_smem_desc(&smem_B[0][0], 32, 3);

// 3. Execute WGMMA
wgmma_m64n64k16_f32_f16_f16(acc, desc_a, desc_b);

// 4. Commit and wait
wgmma_commit_group();
wgmma_wait_group<0>();  // Wait for most recent group
```

### Multiple WGMMAs (Pipelined)
```cuda
wgmma_fence();

// Issue N WGMMAs back-to-back
for (int i = 0; i < N; i++) {
    wgmma_m64n64k16_f32_f16_f16(acc, desc_a[i], desc_b[i]);
    wgmma_commit_group();  // Each WGMMA gets its own group
}

// Wait for all (can wait for specific groups)
wgmma_wait_group<0>();  // Wait for last N groups
```

---

## 💾 SHARED MEMORY LAYOUT

### Optimal Padding (Bank Conflict-Free)
```cuda
// ❌ WRONG: Causes bank conflicts
__shared__ __half smem_A[64][24];  // 24 × 2 = 48 bytes (1.5 banks)

// ✅ CORRECT: Bank conflict-free
__shared__ __half smem_A[64][32];  // 32 × 2 = 64 bytes (2 banks)
```

### Swizzle Modes
```
Mode 0: No swizzle (simple linear addressing)
Mode 1: 32B swizzle  (for small tiles)
Mode 2: 64B swizzle  (for medium tiles)
Mode 3: 128B swizzle (for 64×32+ layouts) ← USE THIS
```

**Why Mode 3?** Eliminates bank conflicts for 64-row tiles with 32+ column padding.

---

## 🔄 MATRIX TRANSPOSE

### For A @ B^T Computation
```cuda
// Load A row-major (standard)
for (int idx = tid; idx < 64 * 16; idx += 256) {
    int row = idx / 16;
    int col = idx % 16;
    smem_A[row][col] = A[row * 16 + col];
}

// Load B TRANSPOSED for B^T
for (int idx = tid; idx < 64 * 16; idx += 256) {
    int row = idx / 16;
    int col = idx % 16;
    smem_B[col][row] = B[row * 16 + col];  // Note: [col][row]
}
```

**Alternative:** Use `.trans` PTX modifier (if supported)

---

## ⚡ PERFORMANCE OPTIMIZATION

### Critical Optimizations
1. **Padding:** 32 elements (not 24) → **+20% perf**
2. **Swizzle mode 3:** 128B swizzle → **+15% perf**
3. **cp.async loads:** Async copies → **+25-30% perf**
4. **Vectorized I/O:** uint4 (16B) → **+10% perf**
5. **TMA (Step 4):** Hardware DMA → **+40-60% perf**

### Expected Performance Progression
```
Step 1 (single):    3-4 TFLOPS     (baseline)
Step 2 (4× tiles):  10-15 TFLOPS   (3-4× scaling)
Step 3 (pipeline):  30-40 TFLOPS   (3× gain from overlap)
Step 4 (TMA):       45-55 TFLOPS   (1.5× gain from TMA)
Step 5 (clusters):  55-65 TFLOPS   (1.2× gain from DSM)
```

---

## 🐛 COMMON PITFALLS

### 1. Wrong Thread Mapping
```cuda
// ❌ WRONG: Linear mapping
const int flat_idx = tid * 32 + i;

// ✅ CORRECT: Warp-aware pattern
const int warp_id = tid / 32;
const int lane = tid % 32;
// ... complex mapping (see above)
```

### 2. Insufficient Padding
```cuda
// ❌ WRONG: Bank conflicts
__shared__ __half smem[64][16];  // or [64][24]

// ✅ CORRECT: Aligned to 64B
__shared__ __half smem[64][32];
```

### 3. Fence Ordering
```cuda
// ❌ WRONG: Descriptors before fence
uint64_t desc = make_smem_desc(...);
wgmma_fence();

// ✅ CORRECT: Fence first
wgmma_fence();
uint64_t desc = make_smem_desc(...);
```

### 4. Missing Transpose
```cuda
// ❌ WRONG: Computes A @ B (not A @ B^T)
smem_B[row][col] = B[row * K + col];

// ✅ CORRECT: Transpose for B^T
smem_B[col][row] = B[row * K + col];
```

---

## 📊 PROFILING COMMANDS

### Register Usage
```bash
nvcc -arch=sm_90a --ptxas-options=-v kernel.cu
# Look for: "X registers", "0 bytes spill" (good)
```

### Bank Conflicts
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./test_wgmma
# Target: 0 conflicts
```

### WGMMA Utilization
```bash
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    ./test_wgmma
# Target: >50% for Step 3+
```

### Full Profiling
```bash
ncu --set full -o profile ./test_wgmma
# Open in Nsight Compute GUI for detailed analysis
```

---

## 🔒 CONSTANT-TIME CHECKLIST

### Must Have (All Steps)
- ✅ No data-dependent branches (`if (data[i] > threshold)`)
- ✅ Fixed iteration counts (no `while` or early `break`)
- ✅ No conditional writes (`if (cond) C[i] = x`)
- ✅ Deterministic register allocation

### Validation
```bash
# SASS analysis (no branches)
cuobjdump -sass kernel.o | grep -E "BRA|BRX|JMP"
# Should be empty or only unconditional branches

# Hardware counters (identical for different inputs)
ncu --metrics sm__cycles_elapsed.avg ./test input1
ncu --metrics sm__cycles_elapsed.avg ./test input2
# Should be identical
```

---

## 📚 ESSENTIAL REFERENCES

1. **PTX ISA 8.3+:** Section 9.7.13 (wgmma instructions)
   - URL: https://docs.nvidia.com/cuda/parallel-thread-execution/

2. **CUTLASS 3.x:** Hopper GEMM examples
   - `examples/48_hopper_warp_specialized_gemm/`
   - GitHub: https://github.com/NVIDIA/cutlass

3. **CUDA Programming Guide:** Tensor Core programming
   - Chapter 7.22: Thread Block Clusters
   - Chapter 7.25: TMA (Tensor Memory Accelerator)

4. **Nsight Compute:** Profiling guide
   - URL: https://docs.nvidia.com/nsight-compute/

---

## 🎯 QUICK VALIDATION

### After Each Change
```bash
# 1. Compile
./build_test_wgmma_corrected.sh

# 2. Check register usage
# Expected: 45-55 registers, 0 bytes spill

# 3. Run test
./build/bin/test_wgmma_corrected

# 4. Validate
# Expected: Max error < 1e-2, 2.8-3.5 TFLOPS (Step 1)

# 5. Profile (optional but recommended)
ncu --set full ./build/bin/test_wgmma_corrected
```

---

## ✅ SUCCESS CRITERIA (Step 1)

- ✅ **Compile:** No errors, 45-55 registers, 0 spills
- ✅ **Correctness:** Max error < 1e-2 vs CPU reference
- ✅ **Performance:** 2.8-3.5 TFLOPS median
- ✅ **Bank conflicts:** 0 (Nsight Compute)
- ✅ **Thread mapping:** Correct (matches reference)

---

**Print this card and keep it next to your keyboard!**

*Quick Reference v1.0 - Expert CUDA Architect - Oct 27, 2025*
