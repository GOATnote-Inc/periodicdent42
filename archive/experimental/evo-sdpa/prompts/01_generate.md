You are **CUDA Kernel Evolution Specialist**. Generate **one** fused SDPA *forward* kernel (Ada, sm_89) for our repo, using **I1** in `00_task.md`. Emphasize diversity of tiling/pipelines; keep to constraints below.

### Constraints (must obey)
- CUDA C++17; optional inline PTX for `cp.async` / `st.global.cs`; **no Triton**.
- Use **tensor cores** (`wmma::mma_sync`), FP32 accum; **store→softmax→rebuild** between QKᵀ and P·V.
- Global mem coalesced; stage Q/K/V tiles into SMEM; prefer **cp.async** (16 B aligned on Ada).
- Causal masking before exp; `<typename T>` where T∈{half, nv_bfloat16}.
- Avoid bank conflicts: pad SMEM rows to multiple of 16 elements when used with WMMA loads.
- Regs/thread ≤64 (≤72 max); include ptxas reg count in comments.

### Initial sketch (free to change if you justify)
- d=64: (M,N,K)=(128,64,64), 4 compute warps, 1 producer warp, 1 epilogue warp
- d=128: (M,N,K)=(64,64,64), 4 compute warps, 2 producer/epilogue warps

### Deliverables
1) `kernels/sdpa_fused.cu` translation unit with:
   - `struct SdpaParams { const void* Q; const void* K; const void* V; void* O; int B,H,L,d; float scale; bool causal; };`
   - `cudaError_t sdpa_fused_forward(const SdpaParams&, cudaStream_t);` dispatcher picking tile/pipeline by (d,L)
2) `kernels/runtime.hpp` minimal helper included by the TU
3) A brief top comment: **Rationale**, **Assumptions**, **Measured risks** (occupancy, cp.async depth, bank conflicts)

### Style
- 2–3 stage cp.async ring with `commit_group` / `wait_group`; `#pragma unroll` inner K loop
- Only use 16B cp.async when aligned; scalar tail fallback
- Keep code **compilable** and **green-first**: correctness over speed

