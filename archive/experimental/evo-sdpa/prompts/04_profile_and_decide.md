You are **CUDA performance triage**. We have v6a (GREEN) and v7a (overlap). Run **Nsight Compute** on these shapes:

- (B=1,H=8,L=512,d=64, causal=0)  — mission
- (B=2,H=8,L=2048,d=64, causal=1) — long
- (B=2,H=8,L=2048,d=128, causal=1) — wide

**Tasks (sequential; do not skip).**
1) Use `bench/profile_ncu_full.py` to collect metrics in `nsight/metrics.txt`. Extract:
   - `sm__pipe_tensor_active` (%), `smsp__inst_executed_pipe_tensor.sum`
   - `smsp__inst_executed_pipe_lsu.sum`, `smsp__gmio_stall.avg.pct`
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`
   - `smsp__warps_active.avg.pct_of_peak_sustained_active`

2) Diagnose bottleneck with a short paragraph for each shape:
   - **Compute‑bound?** (Tensor pipe >70% and DRAM <50%) → overlap won't help; try micro‑tiling or interleaved layout to cut bank conflicts.
   - **Memory‑bound?** (DRAM >70% or gmio_stall high) → deepen pipeline (3‑stage), enlarge producer quota, or redo stage chunking.

3) Emit **I3 bullets** (≤6 lines) to feed into `02_refine_with_insights.md`.
   - Be specific (e.g., "switch L≥2048 to (M,N)=(128,64), 3‑stage; producer = 2 warps; shrink per‑warp scratch by 1/2 to fit ≤64 KiB").

4) Propose **one child plan** (v7b) that modifies exactly two levers (label them), *compatible with WMMA loads*:
   - e.g., **(lever #5)** adopt **interleaved column‑major** SMEM layout for Kᵀ that preserves contiguous columns for `wmma::load_matrix_sync` while breaking bank conflicts; and **(lever #3)** assign warp 6 as **second producer** at L≥2048.

**Return format.**
- A compact report:
  - Per‑shape table of metrics (key 5–6 only)
  - 1‑paragraph diagnosis per shape
  - I3 bullet list (actionable)
  - v7b child plan with the two levers and a 5‑line "why faster"

