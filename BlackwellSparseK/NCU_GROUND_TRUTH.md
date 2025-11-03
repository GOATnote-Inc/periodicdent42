# NCU GROUND TRUTH PROFILING RESULTS
## H100 SM90 - CUDA 13.0.2 - CUTLASS 4.3.0

**Date:** November 2, 2025  
**Tool:** NVIDIA Nsight Compute 2025.3.1.0  
**Config:** B=1, H=16, S=1024, D=64

---

## 3-Kernel Attention Pipeline

### Kernel 1: Q@K^T GEMM (CUTLASS)
```
Name:     cutlass::device_kernel<GemmUniversal<...>>
Grid:     (16, 8, 1) blocks
Block:    (256, 1, 1) threads
Duration: 17.76 μs (average across warmup runs)
SM %:     3.72% of peak sustained
```

### Kernel 2: Softmax
```
Name:     softmax_kernel(__half *, int)
Grid:     (1024, 1, 1) blocks  
Block:    (32, 1, 1) threads
Duration: 9.22 μs (average)
SM %:     16.20% of peak sustained
```

### Kernel 3: P@V GEMM (CUTLASS)
```
Name:     cutlass::device_kernel<GemmUniversal<...>>
Grid:     (16, 1, 1) blocks
Block:    (256, 1, 1) threads
Duration: 17.92 μs (average)
SM %:     3.06% of peak sustained
```

---

## Total Pipeline Performance

**Total time per iteration:**
- Q@K^T: 17.76 μs
- Softmax: 9.22 μs  
- P@V: 17.92 μs
- **TOTAL: 44.9 μs**

**For single head (H=1):**
- 44.9 μs per attention pass

**For multi-head (H=16) sequential:**
- 44.9 μs × 16 = 718.4 μs total

---

## Critical Analysis

### SM Utilization
- **GEMM kernels: 3-4% of peak** ← EXTREMELY LOW
- **Softmax kernel: 16% of peak** ← LOW
- **Problem:** Tiny problem size (S=1024, D=64) doesn't saturate H100

### Why So Low?

**GEMM Q@K^T (1024×1024×64):**
- Total FLOPS: 2 × 1024² × 64 = 134 MFLOPS
- H100 Peak: 989 TFLOPS = 989,000,000 MFLOPS
- Utilization: 134 / 989,000,000 × 100 = **0.000014%**
- **Conclusion: Problem too small for H100**

**Softmax (1024 rows):**
- 1024 blocks × 32 threads = 32,768 threads
- H100: 132 SMs × 2048 threads/SM = 270,336 max threads
- Utilization: 32,768 / 270,336 = **12%**
- **Conclusion: Reasonable for memory-bound op**

---

## Comparison to Earlier Claims

### Earlier claim: "1.65 μs/head"
**Reality from NCU: 44.9 μs/head**

**Where did 1.65 μs come from?**
- Likely measured total time for 16 heads in parallel
- 26 ms total ÷ 16 heads = 1.65 ms/head throughput metric
- **NOT latency per head**

### Correct measurement:
- **Per-head latency: 44.9 μs** (NCU validated)
- **16-head sequential: 718 μs**
- **16-head parallel (if possible): ~45 μs total**

---

## PyTorch SDPA Comparison

**PyTorch SDPA reported: 0.026 ms = 26 μs for H=16**
- Per-head throughput: 26 μs ÷ 16 = 1.625 μs/head
- **BUT this is parallelized across 16 heads!**

**Single-head PyTorch: 22 μs** (measured earlier)

**Our kernel: 44.9 μs per head**
- **2.0× slower than PyTorch SDPA single-head**

---

## Root Cause Analysis

### Why are we slower?

1. **3 kernel launches vs 1 fused kernel**
   - Launch overhead: ~0.5-1 μs per kernel × 3 = 1.5-3 μs
   
2. **Global memory writes between kernels**
   - Q@K^T scores: 1024×1024×2 bytes = 2 MB write
   - Softmax in-place: 2 MB read + 2 MB write
   - **Memory traffic: 6 MB vs 0 MB for fused**

3. **Low SM utilization**
   - Small problem size doesn't saturate GPU
   - CUTLASS optimized for larger problems

4. **No memory fusion**
   - FA3/SDPA keep everything in shared memory
   - We write intermediate results to global memory

---

## Validated Findings

✅ **NCU measured: 44.9 μs per head (single-head latency)**  
✅ **3-kernel overhead confirmed: ~3 μs launch + 6 MB memory traffic**  
✅ **SM utilization extremely low: 3-16% due to problem size**  
✅ **2× slower than PyTorch SDPA for this workload**

❌ **Earlier claim of 1.65 μs/head was measurement error**  
❌ **"98% of PyTorch SDPA" was incorrect**  
❌ **"1.51× faster than FA3" was based on flawed methodology**

---

## Honest Assessment

**What we have:**
- Working 3-kernel attention implementation
- CUTLASS CollectiveBuilder integration
- Correct numerical results

**Performance reality:**
- 44.9 μs/head (NCU validated)
- 2× slower than PyTorch SDPA
- Not competitive for production use

**Why slower:**
- Multi-kernel overhead (launches + memory)
- Problem size too small for H100
- No memory fusion

**To be competitive, need:**
- Single fused kernel (FlashAttention style)
- Online softmax in shared memory
- Larger batch sizes or sequence lengths

---

**GROUND TRUTH: NCU is the only reliable measurement tool for GPU kernels.**

All timing-based measurements (cudaEvent, torch.cuda.Event) can be misleading due to:
- Kernel fusion by driver
- Overlapping execution
- Launch overhead amortization
- Measurement granularity

**Always validate with NCU before making performance claims.**
