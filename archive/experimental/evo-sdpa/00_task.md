# Task: Fused SDPA (Ada, sm_89) — Beat CUTLASS-style baselines

**Objective.** A single-pass, numerically-stable fused SDPA:
O = softmax(Q·Kᵀ / √d)·V with causal & optional dropout. Fuse QKᵀ, streaming softmax, and P·V to avoid HBM materialization.

**Correctness gates.**
- ≤1e-3 abs/rel vs PyTorch SDPA (FP16/BF16) with FP32 accum
- Handles: causal flag, variable L, padding; d∈{64,80,96,128}; L∈[128,8192]
- Deterministic dropout path (when enabled)

**Targets.**
- ≥2× PyTorch SDPA and **beat CUTLASS GEMM+softmax piping** on mission shapes
- Stretch: ≥60× vs naïve unfused for long L, big B×H

**Hardware/compile.**
- RTX 4090 (sm_89, 24 GB, ~1008 GB/s), CUDA 12.4.1, PyTorch 2.4.0, Python 3.11
- Flags: `-O3 -arch=sm_89 --use_fast_math -lineinfo -Xptxas -v`
- Average over 100 runs; TF32 allowed for FP32 paths

**Threadblock & budget.**
- Grid: (ceil_div(L, M_tile), B*H); block: 128–256 threads (4–8 warps)
- SMEM ≤ 64 KiB/CTA (aim 48–64 KiB), regs ≤ 64/thread initial (≤72 max)

**Current baseline & lineage.**
- v6a: WMMA QKᵀ + store→softmax→rebuild + WMMA PV; per‑warp scratch; **GREEN 100%**; ~1177 µs
- v7a: cp.async overlap (producer warp, 2–3 stages); **GREEN**; ~1162 µs; minimal speedup → profile first

**Design invariants.**
- One warp owns 16 rows (mᵢ, lᵢ) for streaming softmax
- FP32 accum everywhere; cast O at epilogue
- Legal cp.async (Ada 16B minimum preferred; fall back to 8/4 only if supported)
