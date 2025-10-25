# Fused FlashAttention: Online Softmax and WMMA on Ada (L4, sm_89)

## FlashAttention-2 and the Online Softmax Trick

FlashAttention v2 uses an "online softmax" algorithm to tile attention computation without ever materializing the full score or probability matrices in memory[^1]. In essence, we partition the $K$ and $V$ matrices into blocks and compute attention one block at a time, while maintaining running statistics for softmax. After processing all blocks, we obtain the exact same softmax output as the naive approach but with far fewer memory operations[^1]. 

This is achieved by updating a running row-wise max (`m`) and sum of exponentials (`l`) per query as each block of $K$ is processed. Partial outputs are rescaled whenever a new larger max is found, ensuring numerical stability (safe softmax) without storing intermediate results[^2][^3]. By computing attention block-by-block and normalizing on the fly, we avoid expensive memory reads/writes of the intermediate matrices $S$ (scores) and $P$ (softmax probabilities)[^1]. This fusion is the key to FlashAttention's speedup: it trades a bit of extra computation (warp reductions and exponentials each tile) for dramatically reduced memory traffic[^1].

### How it works

After each partial multiplication $Q \times K^T$ (for one tile of keys of width e.g. 32 or 64), the thread block computes the local softmax contribution entirely in registers. Each warp holds a $16\times16$ fragment of the partial score matrix `c_frag` in FP32. The warp first finds the max score for each row in its fragment using warp-level reduction (with `__shfl_sync`)[^4]. These local maxima are compared with the global max `m_smem` for those rows (from previous tiles), and if larger, we update the global max and rescale the accumulated partial output $U$ accordingly[^5][^6]. 

Then the warp computes the sum of $\exp(\text{score} - m_{\text{new}})$ for its fragment (only valid columns) and accumulates that into the global sum `l_smem` for each row[^7][^8]. At this point, the global `m_smem[r]` and `l_smem[r]` represent the softmax normalization factors for the row after processing the current tile. 

Finally, the warp writes the exp-normalized probabilities for its fragment into shared memory `sP`[^9]. Once all warps have processed the tile and written their portion of `sP` (hence a block-wide `__syncthreads()` to make `sP` visible)[^10], the block immediately performs the WMMA multiplication $P \times V$ for that tile, accumulating into the output buffer $U$[^11]. 

This sequence is repeated for each $K$/$V$ tile. After the last tile, each row's `m_smem` and `l_smem` hold the final max and normalization denominator; the output $U$ in shared memory is then normalized by `l_smem` and written to global memory as the final result[^12]. 

**In summary**: We fuse Q@K, softmax, and P@V all in one kernel, never storing the full $S$ or $P$ matrices, exactly as FlashAttention describes[^1].

---

## Leveraging Tensor Cores with WMMA (16×16×16)

To reach our <40 μs goal, we heavily utilize Tensor Cores via CUDA's WMMA API. The kernel operates on tiles of size 16×16 (M×N) with K=16 (the "warp-matmul" shape), using half-precision inputs and accumulating in FP32 for accuracy[^13]. We choose tile dimensions that are multiples of 16 to match WMMA requirements (e.g. $D_{\text{head}}=64$ so we iterate over 4 chunks of K=16)[^14]. 

### CTA and Warp Layout

Each CTA (thread block) computes a $T_M\times T_N$ output tile (e.g. 32×32 or 64×64) corresponding to a block of queries and keys. In our current design, we use `TILE_M = 32`, `TILE_N = 32` (so each block covers 32 query positions by 32 key positions). With 32 threads per warp, this means 4 warps per block arranged in a 2×2 grid to cover the 32×32 tile[^15]. 

For example:
- Warp 0 computes the 16×16 sub-tile for rows 0–15, cols 0–15
- Warp 1 handles rows 0–15, cols 16–31
- Warp 2 handles rows 16–31, cols 0–15
- Warp 3 handles rows 16–31, cols 16–31[^16]

This mapping ensures each warp computes a disjoint part of the tile.

### WMMA Usage Best Practices

**WMMA usage**: We load fragments from shared memory using `wmma::load_matrix_sync`, multiply-accumulate with `wmma::mma_sync`, and use `wmma::store_matrix_sync` to write results[^17][^18]. 

**One crucial best practice** is to use row-major layout for $Q$ and $P$ fragments, and col-major for $K^T$ fragments, to match the WMMA API expectations. Our shared memory is organized accordingly: we store $Q$ and $P$ as row-major `[TILE_M][D_PAD]` and `[TILE_M][TILE_N]`, but store $K^T$ as `[TILE_N][D_PAD]` so that threads can fetch contiguous elements along the K dimension for the B fragment[^19][^20]. 

We also pad dimensions (e.g. `D_PAD=64`) to be multiples of 16 for alignment[^21].

### Fragment Layout and LUT

Each warp's accumulator fragment (`wmma::fragment<accumulator,16,16,16,float>`) contains 16×16 = 256 results, distributed across the 32 lanes. For FP16->FP32 WMMA on Ampere/Ada, each lane holds 8 elements of the fragment[^22]. The layout is fixed but not trivial – the values are interleaved by lane. 

To handle this, our code includes a lookup table `WMMA_ACCUM_LUT[lane][elem] = (row, col)` that tells each thread which fragment element corresponds to which matrix cell[^23]. Using this LUT, we can map each thread's 8 partial scores to their proper row and column in the 16×16 tile. This is critical for the fused softmax: e.g. to find the max of a row in the 16×16 fragment, each lane examines its `c_frag.x[i]` values, uses the LUT to find the global (r, c) indices, and only considers those belonging to the current row `r`[^24]. 

This approach lets us perform row-wise reductions and updates correctly even though the data is fragmented across warps and lanes. (NVIDIA doesn't publicly document the exact fragment index mapping, so using a precalculated LUT based on experimentation or the wmma pipeline's known behavior is a common trick.)

### Accumulation Strategy

We accumulate in FP32 to avoid precision loss during softmax. We also initialize the accumulator with zeros via `wmma::fill_fragment(c_frag, 0.0f)`[^25] for each tile. 

Another best practice is to ensure no register reuse conflicts for fragments – i.e. we allocate separate fragments for the Q@K computation and for the P@V computation, or reuse only after finishing the first phase, to avoid overwriting values needed for softmax. 

In our fused kernel, once we finish using `c_frag` (QK results) to compute softmax and write out `sP`, we then proceed to load fragments for $P \times V$. We perform a second set of WMMA operations: load `a_frag` from `sP` and `b_frag` from `sV` (with appropriate tiling) and accumulate into a new `c_frag_pv` for the output[^26]. This produces the partial output matrix $U$ (unnormalized attended values). 

Each warp again writes its 16×16 result to shared memory (we use a scratch buffer `sPV_frag` per warp to avoid bank conflicts) and then add it into the global output `U_smem`[^27][^28]. 

**By using WMMA for both Q@K^T and P@V phases, we leverage Tensor Cores for almost all arithmetic** – significantly boosting throughput.

---

## Memory Movement: cp.async and Multi-Stage Pipelining

Efficient memory hierarchy usage is crucial. We load $Q$, $K$, and $V$ from global memory as needed for each tile, and we aim to overlap these transfers with computation using `cp.async` (CUDA's asynchronous copy pipeline). On NVIDIA Ada (sm_89, as in L4 GPUs), `cp.async` allows copying data from global memory into shared memory in the background, without stalling warps, and then synchronizing at a later point.

### Double/Triple Buffering

Our fused kernel employs a double-buffered (and optionally triple-buffered) pipeline for $K$ and $V$ tiles. We allocate multiple shared-memory buffers `sK_u8` and `sV_u8` (each `[NUM_STAGES][TILE_N][D_PAD]`) to stage the incoming data[^29]. 

For example, with 2-stage (double) buffering, while the current tile's $QK^T$ and softmax are being computed, the next tile's $K$ and $V$ bytes are already being fetched via `cp.async` into the alternate buffer. 

Concretely, each iteration of the tile loop does something like:
1. Issue async copy for tile `t+1` (except on the last iteration) and call `__pipeline_commit()`
2. Wait for tile `t` data to be ready via `__pipeline_wait_prior(0)` (for 2-stage) or `wait_prior(1)` (for 3-stage) once we've allowed one tile to be outstanding[^30]
3. Convert the just-arrived tile `t` data from `uint8` to `half` and store in the working buffers `sKT` and `sV` with proper layout[^31][^32]
4. Proceed to the WMMA compute on tile `t`

This design overlaps the global memory latency of tile `t+1` with the computation on tile `t`. By the time we finish computing tile `t`, the next tile's data is already in shared memory, ready to go.

### Pipeline Intrinsics

We use the new CUDA 12 pipeline intrinsics for clarity: e.g. `__pipeline_memcpy_async(dst, src, 128)` (which under the hood issues a `cp.async.cg.shared.global.L2::128B` for a 128-byte chunk) and `__pipeline_wait_prior(n)` to wait for all but `n` previous copy groups[^33]. 

**A full `__syncthreads()` is done after `wait_prior`** because while `wait_prior` ensures all threads have their data, it doesn't globally synchronize the threads' execution[^34]. 

We also align and coalesce these async copies for efficiency: each warp copies 16 bytes per instruction, and we group warps to cover 128B cache lines from global memory. (The code uses vectorized loads of `uint4` and a lane-group stride to coalesce 16 bytes across 4 lanes[^35][^36]).

### Stage Selection

In our specific implementation, we default to 2-stage pipeline for sequence length $S=512$ (since latency is relatively small), but a 3-stage pipeline can be enabled for much longer sequences[^37][^38]. The 3-stage mode allocates an extra buffer (3 buffers total) and prefetches two tiles ahead, using `__pipeline_wait_prior(1)` to allow one outstanding transfer to overlap with computation[^39][^40]. 

This can further hide latency for very deep sequences at the cost of more shared memory usage. The general heuristic is to choose the number of stages such that `(NUM_STAGES-1) * tile_load_latency ≈ tile_compute_time` for optimal overlap.

### Avoiding Bank Conflicts

We also take care to optimize shared memory accesses. One trick used is an XOR swizzle when storing $K^T$ and $V$ in shared memory[^41]. By XOR'ing the column index for certain rows (e.g. `d_swz = d ^ ((n & 1) * 8)` toggling an offset for odd rows)[^42], we spread accesses from different threads across different banks, mitigating systematic bank conflicts when all threads access the same column modulo 32. 

This XOR swizzle has no effect on correctness (it just permutes where data is stored in SMEM) but can improve memory access parallelism when reading/writing in patterns like dequantization loops[^43]. We similarly ensure alignment (`alignas(16)`) of shared arrays and use broadcasting where possible to fully utilize memory buses (the code loads values in 128B chunks whenever it can)[^44].

---

## Warp-Level Reductions and Synchronization

Because the softmax is done in-tile, we rely on warp-level operations for efficiency. Each warp uses shuffle intrinsics to compute reductions across its 32 lanes. Our code defines `warp_reduce_max(float v)` and `warp_reduce_sum(float v)` that use the typical tree-reduction pattern with `__shfl_down_sync` to propagate partial results[^45]. 

After these calls, lane 0 in the warp holds the reduction result (max or sum), which we then broadcast to all lanes with `__shfl_sync(..., src_lane=0)`[^46]. 

### Row-wise Reductions

For example, to find the max of a row in the 16×16 fragment, each lane examines its 8 scores for that row and computes a local max, then we do `warp_reduce_max` so that ultimately every lane gets the row's max in a register[^47][^48]. This value (call it `m_tile[r]`) is then compared to the global max in `m_smem[r]` and updated by lane 0 if needed[^49][^50]. 

We perform a similar warp reduction for the exponential sums (`l_add`) per row[^51]. Using warp shuffles avoids the need for slower atomic operations or extra shared memory for these 16-wide reductions – it's all done within the warp in just a few instructions.

### Synchronization Points

**Synchronization**: Within a warp, `__shfl_sync` ensures threads stay in lockstep for these reductions (all threads in the warp participate by design). But we also need to coordinate across warps. In the fused softmax algorithm, after each warp updates the global `m_smem` and `l_smem` for its rows, we issue a `__syncwarp()` (which is mostly precautionary here since each warp handles different rows) followed by a block-wide `__syncthreads()`[^52][^53]. 

The block sync is critical: it ensures that when warps proceed to write out their portion of `sP` (the softmax-ed probabilities) and then do the WMMA $P \times V$, they see the finalized normalization values from all warps. In other words, one warp might update `m_smem[0]` for row 0, and another warp needs that value before writing its own `sP[0][*]`. The `__syncthreads()` after updating `m_smem`/`l_smem` serves as the barrier to make the new `m_new` visible to all[^54]. 

Likewise, after writing `sP`, we sync again before the $P \times V$ multiply so that no warp starts loading incomplete data from `sP`[^55]. These synchronization points are the only places we incur warp divergence penalties, but they are necessary for correctness of the fused softmax.

### Zeroing Strategy

We also note that our implementation pre-zeros the entire 16×16 portion of `sP` for each warp before writing any new values[^56]. This avoids a subtle bug where leftover data from a previous tile (especially when a tile has fewer than 16 columns remaining) could pollute the next calculation. We parallelize that zeroing by having each lane set a few elements, and use `__syncwarp()` to ensure the warp's tile is clean before proceeding[^57][^58].

---

## Tile Size (32×32 vs 64×64) and Resource Trade-offs

Our chosen tile size of 32×32 (with D=64) is a balance between per-thread work and occupancy. A larger tile (64×64) could in theory reduce the number of iterations over K and improve compute intensity – in fact, the CUTLASS-based xFormers attention uses a 64×64 tile on this same problem and achieves ~33 μs latency as a result. 

However, **larger tiles consume more registers and shared memory per block**, which can drastically lower occupancy[^59]. In one profile, the 64×64 WMMA kernel was limited by registers to only 4 blocks per SM (theoretical occupancy ~33%), and achieved under 10% occupancy in practice[^60]. 

The benefit is that each warp does more work with tensor cores, so fewer warps are needed – xFormers chooses to maximize work per warp at the cost of having fewer warps, and indeed still outperforms smaller-tile kernels[^61][^62].

### Our Approach

Our approach in this phase leans toward the conservative 32×32 tile to keep occupancy higher (more warps active to hide latency). With `TILE_M=N=32`, we use 4 warps/block and our shared memory usage is moderate (~35 KB in Stage-2, growing to ~39 KB in Stage-3 with buffers) – this allows ~5 blocks per SM limited by SMEM, or potentially more if registers are below the limit[^63]. 

In contrast, a 64×64 tile with 8 warps might use ~60+ KB SMEM and also more registers, often limiting to 4 blocks/SM or fewer[^64][^65].

### Register Pressure

**Register pressure is a major factor here**: with many intermediate values (for softmax, dequantization, etc.), registers per thread can grow quickly. We target ≤128 registers per thread for the fused kernel[^66]. Staying within this budget should allow at least 4 warps * 4 blocks = 16 warps/SM (one block per scheduler on L4), which is roughly 30% theoretical occupancy. 

If we greatly exceed 128 registers, occupancy could drop further (e.g. the 64×64 kernel used so many registers it only allowed ~4 warps active out of 48, as NCU showed[^67]).

### The Trade-off

The lesson from NVIDIA's and Meta's engineers is that **sometimes lower occupancy is acceptable if each warp does a lot more useful work**[^68][^69]. In our case, we will monitor register usage carefully and perhaps experiment with tile sizes in later phases. 

- A smaller tile (e.g. 32×32) means more tile iterations (16 tiles for 512 columns instead of 8 for a 64×64), but it may improve latency hiding
- A larger tile (64×64) means fewer iterations and less loop overhead, but one must ensure the GPU isn't starved of warps

Finally, note that $D=64$ is conveniently a multiple of 16, so our WMMA K-dimension fits exactly. This avoids any wasted computation on padded values for these kernels. If $D$ were not a multiple of 16, we'd pad to the next multiple (as `D_PAD`) so that the wmma fragment is fully utilized.

---

## Nsight Compute Metrics to Track

To ensure the kernel is optimized, we will rely on profiling metrics from Nsight Compute (NCU):

### 1. Occupancy

Check both **theoretical occupancy** (based on registers and SMEM per block) and **achieved occupancy** (average active warps per cycle). We want to confirm that our register usage doesn't cripple occupancy. For example, in a reference 64×64 kernel, theoretical occupancy was ~33% but achieved only ~9%[^70] due to only ~4.5 warps active per SM. 

Our 32×32 kernel should see higher achieved occupancy since more blocks can reside per SM (e.g. SMEM allowed ~5 blocks[^71]). If achieved occupancy is very low, warps may not be able to hide memory latency, leading to idle cycles[^72]. 

NCU's **Active Warps per SM** and **Issue Slot Utilization** are good indicators here (we want a high percentage of issue slots used, meaning the GPU isn't often idle waiting for work)[^73][^74].

### 2. Warp Stall Reasons

We'll look at the breakdown of why warps are not issuing. If we see a lot of "**Stall due to Memory Dependency**" or long scoreboard stalls, that might indicate insufficient pipelining (e.g. not enough `cp.async` stages to hide latency). In that case, increasing `NUM_STAGES` to 3 or optimizing the `cp.async` scheduling could help. 

Conversely, if "**Execution Dependency**" or "**Not Selected**" (no eligible warp) is high, that could point to low occupancy or synchronization bottlenecks[^75]. Since we deliberately have a few `__syncthreads()`, those will show up as periods where warps are waiting; this is expected, but if it dominates, that's an issue.

### 3. Memory Throughput

We will verify that global memory loads are efficient. Metrics like **L2 and DRAM throughput** or **%peak** can tell us if we are memory-bound. In this workload, we expect to be more compute-bound than memory-bound – prior analysis showed DRAM utilization under 10% for a highly-optimized attention kernel[^76]. 

If our kernel is working correctly, the async pipeline should keep global memory requests in flight while computation proceeds, resulting in high overlap. Low memory utilization plus low compute utilization would hint at some other bottleneck (like latency or occupancy). 

We should also check for **shared memory bank conflicts** (Nsight can report bank conflict counts). Our XOR swizzle should reduce conflicts for K/V loads[^77], but we can confirm by comparing metrics with and without `USE_SMEM_SWIZZLE_XOR`. Ideally, most shared memory transactions should be 128-bit and conflict-free.

### 4. Tensor Core Utilization

We'll examine the number of **HMMA operations issued** and the **achieved TFLOPs**. Metrics such as `sm__inst_executed_pipe_tensor` or the derived FLOP count can tell us if we're nearing hardware limits. On L4 (Ada), peak FP16 TFLOP is very high; we likely won't hit 90%+ of peak due to our extra overhead (softmax work, memory transfers). 

However, we can compare against a known baseline (the xFormers kernel or CUTLASS GEMM) – e.g. if that achieves ~150 TFLOPs (93% of peak)[^78], we'd like to approach similar territory. We should also ensure that our mix of instructions isn't causing pipeline stalls – e.g. tensor cores should be kept busy. If we see low `pipe_tensor` utilization and high `pipe_fp64` or others, something is off.

### 5. Register and Spill Metrics

We will confirm through the **PTXAS info** (captured via `torch.utils.cpp_extension.load` or nvcc output) that register usage per thread stays ≤128 and there are 0 spilled registers to local memory[^79][^80]. NCU can also show spill count. Spills would severely hurt performance (memory loads/stores inside the kernel), so avoiding them is critical given our tight latency target.

### 6. Accuracy Checks

Although not a performance metric, we will validate that the fused softmax produces identical results to the unfused baseline (within FP16/FP32 error). We'll run the kernel on test cases and ensure max error is within tolerance (Stage-3 should ideally be bitwise identical to Stage-2 for the same inputs)[^81].

---

## Summary and Next Steps

By monitoring these metrics, we can iteratively refine the kernel. For instance:
- If **occupancy is low** due to register usage, we might try unrolling less or using more shared memory for interim buffers (trading registers for SMEM)[^82]
- If **memory pipe is under-utilized**, maybe increase stages or tile size to use more bandwidth
- If **tensor pipe isn't saturated**, perhaps we can increase tile dimensions or concurrency

The goal is to ensure our fused kernel on GCP's L4 (Ada) is **well-balanced** – keeping the tensor cores fed with data without starving, and overlapping memory operations so that both the MEM pipeline and the TC (tensor core) pipeline are busy most of the time. 

Achieving this balance will be key to hitting the **~40 μs stretch goal** for B=1, H=8, S=512, D=64. 

Overall, by combining the FlashAttention "online softmax" algorithm with careful WMMA usage, async memory pipelines, and warp-level primitives, we aim to dramatically accelerate the attention mechanism. The notes above (with cited code patterns and literature) will guide the Phase 1 design and Phase 2 implementation of our `flashcore_fused_wmma.cu` kernel. 

The focus will be on preserving correctness while aggressively optimizing for Ada's architecture – using on-chip resources (registers/SMEM) to save off-chip bandwidth, and exploiting the massive compute throughput of tensor cores, all while avoiding the common pitfalls indicated by profiling metrics[^83][^84]. 

**(We have the advantage of standing on the shoulders of prior art like xFormers and FlashAttention – many of their optimizations are informing our approach.)**

---

## Citations

[^1]: FlashAttention-2 paper (tridao.me)
[^2]: [STAGE3_FUSION_FULL_PLAN.md#L138-L146](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L138-L146)
[^3]: [STAGE3_FUSION_FULL_PLAN.md#L42-L50](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L42-L50)
[^4]: [sdpa_fp8_stage_c_wmma.cu#L891-L898](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L891-L898)
[^5]: [sdpa_fp8_stage_c_wmma.cu#L532-L541](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L532-L541)
[^6]: [sdpa_fp8_stage_c_wmma.cu#L528-L536](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L528-L536)
[^7]: [sdpa_fp8_stage_c_wmma.cu#L537-L546](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L537-L546)
[^8]: [STAGE3_FUSION_FULL_PLAN.md#L139-L146](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L139-L146)
[^9]: [sdpa_fp8_stage_c_wmma.cu#L559-L567](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L559-L567)
[^10]: [sdpa_fp8_stage_c_wmma.cu#L570-L573](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L570-L573)
[^11]: [STAGE3_FUSION_FULL_PLAN.md#L46-L50](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L46-L50)
[^12]: [sdpa_fp8_stage_c_wmma.cu#L435-L443](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L435-L443)
[^13]: [sdpa_fp8_stage_c_wmma.cu#L2-L10](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L2-L10)
[^14]: [sdpa_fp8_stage_c_wmma.cu#L422-L430](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L422-L430)
[^15]: [sdpa_fp8_stage_c_wmma.cu#L446-L455](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L446-L455)
[^16]: [sdpa_fp8_stage_c_wmma.cu#L458-L465](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L458-L465)
[^17]: [sdpa_fp8_stage_c_wmma.cu#L118-L126](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L118-L126)
[^18]: [sdpa_fp8_stage_c_wmma.cu#L450-L458](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L450-L458)
[^19]: [sdpa_fp8_stage_c_wmma.cu#L49-L57](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L49-L57)
[^20]: [sdpa_fp8_stage_c_wmma.cu#L860-L868](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L860-L868)
[^21]: [sdpa_fp8_stage_c_wmma.cu#L883-L891](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L883-L891)
[^22]: [sdpa_fp8_stage_c_wmma.cu#L438-L446](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L438-L446)
[^23]: [STAGE3_FUSION_FULL_PLAN.md#L148-L157](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L148-L157)
[^24]: [sdpa_fp8_stage_c_wmma.cu#L708-L717](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L708-L717)
[^25]: [sdpa_fp8_stage_c_wmma.cu#L719-L723](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L719-L723)
[^26]: [sdpa_fp8_stage_c_wmma.cu#L124-L133](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L124-L133)
[^27]: [sdpa_fp8_stage_c_wmma.cu#L299-L307](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L299-L307)
[^28]: [sdpa_fp8_stage_c_wmma.cu#L318-L327](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L318-L327)
[^29]: [sdpa_fp8_stage_c_wmma.cu#L337-L345](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L337-L345)
[^30]: [spatters.ca - Tensor Core matmul on Ada](https://www.spatters.ca/mma-matmul)
[^31]: [sdpa_fp8_stage_c_wmma.cu#L324-L332](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L324-L332)
[^32]: [STAGE3_FUSION_FULL_PLAN.md#L12-L19](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L12-L19)
[^33]: [STAGE3_FUSION_FULL_PLAN.md#L14-L22](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L14-L22)
[^34]: [spatters.ca - Tensor Core matmul on Ada](https://www.spatters.ca/mma-matmul)
[^35]: [STAGE3_FUSION_FULL_PLAN.md#L40-L48](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L40-L48)
[^36]: [STAGE3_FUSION_FULL_PLAN.md#L82-L91](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L82-L91)
[^37]: [STAGE3_FUSION_FULL_PLAN.md#L86-L94](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L86-L94)
[^38]: [STAGE3_FUSION_FULL_PLAN.md#L86-L95](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L86-L95)
[^39]: [sdpa_fp8_stage_c_wmma.cu#L63-L71](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L63-L71)
[^40]: [sdpa_fp8_stage_c_wmma.cu#L516-L524](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L516-L524)
[^41]: [sdpa_fp8_stage_c_wmma.cu#L549-L553](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L549-L553)
[^42]: [sdpa_fp8_stage_c_wmma.cu#L915-L924](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L915-L924)
[^43]: [sdpa_fp8_stage_c_wmma.cu#L554-L558](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L554-L558)
[^44]: [sdpa_fp8_stage_c_wmma.cu#L569-L573](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L569-L573)
[^45]: [sdpa_fp8_stage_c_wmma.cu#L491-L499](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L491-L499)
[^46]: [sdpa_fp8_stage_c_wmma.cu#L500-L503](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu#L500-L503)
[^47]: [NCU_CRITICAL_FINDING.md#L24-L32](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L24-L32)
[^48]: [NCU_CRITICAL_FINDING.md#L13-L21](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L13-L21)
[^49]: [NCU_CRITICAL_FINDING.md#L130-L138](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L130-L138)
[^50]: [NCU_CRITICAL_FINDING.md#L132-L140](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L132-L140)
[^51]: [NCU_CRITICAL_FINDING.md#L67-L75](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L67-L75)
[^52]: [STAGE3_FUSION_FULL_PLAN.md#L211-L219](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L211-L219)
[^53]: [NCU_CRITICAL_FINDING.md#L14-L21](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L14-L21)
[^54]: [NCU_CRITICAL_FINDING.md#L110-L118](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L110-L118)
[^55]: [NCU_CRITICAL_FINDING.md#L38-L46](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L38-L46)
[^56]: [NCU_CRITICAL_FINDING.md#L14-L19](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L14-L19)
[^57]: [NCU_CRITICAL_FINDING.md#L40-L48](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L40-L48)
[^58]: [NCU_CRITICAL_FINDING.md#L52-L60](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L52-L60)
[^59]: [NCU_CRITICAL_FINDING.md#L102-L105](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md#L102-L105)
[^60]: [NCU_CRITICAL_FINDING.md - occupancy](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^61]: [spatters.ca - Tensor Core matmul on Ada](https://www.spatters.ca/mma-matmul)
[^62]: [STAGE3_FUSION_FULL_PLAN.md#L242-L244](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L242-L244)
[^63]: [STAGE3_FUSION_FULL_PLAN.md#L253-L258](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L253-L258)
[^64]: [STAGE3_FUSION_FULL_PLAN.md#L216-L220](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L216-L220)
[^65]: [STAGE3_FUSION_FULL_PLAN.md#L26-L34](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md#L26-L34)
[^66]: [STAGE3_FUSION_FULL_PLAN.md - register budget](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md)
[^67]: [NCU_CRITICAL_FINDING.md - register pressure](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^68]: [STAGE3_FUSION_FULL_PLAN.md - occupancy trade-offs](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md)
[^69]: [NCU_CRITICAL_FINDING.md - work per warp](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^70]: [NCU_CRITICAL_FINDING.md - occupancy metrics](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^71]: [STAGE3_FUSION_FULL_PLAN.md - SMEM limits](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md)
[^72]: [NCU_CRITICAL_FINDING.md - latency hiding](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^73]: [NCU_CRITICAL_FINDING.md - active warps](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^74]: [NCU_CRITICAL_FINDING.md - issue slots](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^75]: [NCU_CRITICAL_FINDING.md - stall reasons](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^76]: [NCU_CRITICAL_FINDING.md - DRAM utilization](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^77]: [sdpa_fp8_stage_c_wmma.cu - XOR swizzle](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu)
[^78]: [spatters.ca - TFLOPs on Ada](https://www.spatters.ca/mma-matmul)
[^79]: [STAGE3_FUSION_FULL_PLAN.md - PTXAS info](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md)
[^80]: [NCU_CRITICAL_FINDING.md - spills](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^81]: [STAGE3_FUSION_FULL_PLAN.md - correctness checks](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md)
[^82]: [STAGE3_FUSION_FULL_PLAN.md - optimization strategies](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md)
[^83]: [NCU_CRITICAL_FINDING.md - profiling guidance](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/NCU_CRITICAL_FINDING.md)
[^84]: [STAGE3_FUSION_FULL_PLAN.md - Ada optimization](https://github.com/GOATnote-Inc/periodicdent42/blob/0df41dd07bea01d092f02f0cf3ea95172f86434a/STAGE3_FUSION_FULL_PLAN.md)

