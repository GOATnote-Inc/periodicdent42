# Task: Fused Scaled Dot-Product Attention (SDPA) kernel, Ada (sm_89)

**Objective:** Implement and tune a **single-pass, numerically-stable fused SDPA**:
```
O = softmax(Q · K^T / sqrt(d)) · V
```
with optional causal masking and dropout. **Fuse** QK^T, streaming softmax, and *Softmax·V* to avoid materializing logits or probabilities to HBM.

**Correctness gates:**
- Relative/abs error vs PyTorch `scaled_dot_product_attention` ≤ 1e-3 (FP16/BF16) with FP32 accum.
- Handles: causal flag, variable seq lengths, padding mask; d_head ∈ {64, 80, 96, 128}; L ∈ [128, 8192]; batch×head up to GPU memory.
- Deterministic RNG path for dropout when enabled.

**Performance target:** ≥ 2× PyTorch reference and **beat CUTLASS GEMM+softmax piping** on the same shapes. Stretch: explore **≥60× vs naïve unfused** reference for long L, large B×H. (Realistic speedups depend on L, d, and memory BW ceilings.)

**Mission Critical Target (from project roadmap):**
- **< 5 μs** for B=1, H=8, S=512, D=64 (current SDPA baseline: 25.94 μs)
- This requires **5.2× faster than PyTorch SDPA** (not just 2×)
- Standing on giants' shoulders (SDPA) to see further

**Hardware/compilation:**
- GPU: L4 (Ada, sm_89, 24 GB GDDR6, ~300 GB/s) - our actual test hardware
- Alternative: RTX 4090 (Ada, 16,384 CUDA cores, 24 GB GDDR6X, ~1008 GB/s)
- CUDA 12.4.1, PyTorch 2.1.0 (our environment)
- Compilation: `-O3 -arch=sm_89 --use_fast_math -lineinfo`
- Prefer **cp.async** multi-stage prefetch; **ldmatrix** + **mma.sync** tensor core paths (TF32/FP16/BF16), FP32 accum.
- Shared memory budget target: ≤ 48–64 KiB/CTA to allow ≥2 CTAs/SM; registers ≤ 64 per thread initially.

**Threadblock mapping (initial guess):**
- CTA owns (M_tile × N_stream) rows of Q and a streaming panel of K,V (N_tile) with 2–3 stage cp.async pipeline.
- Example: (M_tile,N_tile,K_tile) = (128, 64, 64) for d_head=64; adjust for 128.

**Numerical stability:**
- Maintain row-wise `m_i` (max) and `l_i` (sum of exp) as you stream K panels; update with the log-sum-exp trick.
- Accumulate **S·V** in FP32 across K-panels; write O in FP16/BF16.

**Launch policy:**
```
grid = (ceil_div(L, M_tile), B*H) ; blockDim = 128–256 threads (4–8 warps).
```

**Context from previous work:**
- We've achieved 1.25× speedup with partial WMMA (Q@K^T only)
- Current: 1274 μs → Target: < 5 μs = **255× more speedup needed**
- Key insight: Need FULL fusion (not just Q@K^T), streaming softmax, and optimal memory access patterns
- Production libraries (xFormers: 24.22 μs) are the baseline to exceed

