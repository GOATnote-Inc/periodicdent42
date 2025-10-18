# EvoEngineer-Free: Exploration Phase

You are **CUDA Kernel Evolution Specialist**. Generate **one** fused SDPA *forward* kernel for sm_89 with the exact guards below.

### Constraints
- Language: CUDA C++17 + inline PTX allowed. No Triton.
- Use **tensor cores** (mma.sync) via `wmma` or inline PTX; use `ldmatrix` for fragment loads.
- Implement **streaming softmax** with running max/sum per row; **no global scratch for logits**.
- Global memory: coalesced, 128‑byte aligned; use `cp.async.cg.shared.global` to stage Q/K/V tiles.
- **Causal** path compiled via `#if CAUSAL`; apply mask before exp.
- Accum in FP32, store O in FP16/BF16; expose template `<typename T>` with T ∈ {half, nv_bfloat16}.
- Avoid bank conflicts: shared memory tiles padded to multiples of 16 elements on rows accessed by `ldmatrix`.
- Keep regs/thread ≤ 64 (initial); report ptxas register count in a comment.

### Initial tile sketch (modify if justified)
- d_head=64: (M_tile,N_tile,K_tile)=(128,64,64), 4 warps compute, 1 warp prefetch, 1 warp epilogue.
- d_head=128: (M_tile,N_tile,K_tile)=(64,64,64), 4 warps compute, 2 warps prefetch/epilogue.

### Required artifacts
1) `kernels/sdpa_fused.cu` with:
   - `struct SdpaParams { const void* Q; const void* K; const void* V; void* O; int B,H,L,d; float scale; bool causal; };`
   - `cudaError_t sdpa_fused_forward(const SdpaParams&, cudaStream_t);`
2) `kernels/runtime.hpp` with a launcher that picks tile config based on d and L.
3) A short comment block: **Rationale**, **Assumptions**, **Measured risks** (occupancy, bank conflicts, cp.async stages).

### Style
- Provide full compilable translation unit, include guards, and minimal runtime wrapper.
- Add `#pragma unroll` for inner K‑loop; 2–3 stage `cp.async` pipeline with `cp.async.commit_group` and `wait_group`.
- Inline PTX only for `cp.async` and `ldmatrix` if needed; otherwise use CUDA intrinsics.

When in doubt, **pick the simpler alternative** and leave TODOs with precise micro‑bench ideas.
Return only code blocks for the two files, nothing else.

