# Building AI Agents That Consistently Deliver Expert-Level CUDA Kernels

> _"The GPU is a throughput machine. Kernels win when memory and math are choreographed with mechanical sympathy for the hardware."_

This whitepaper expands the 2023–2025 best practices for AI agents that author and optimize CUDA kernels on NVIDIA Ampere (SM80) and Hopper (SM90) GPUs. It distills modern kernel-engineering doctrine into actionable playbooks, tooling stacks, and architectural templates that we teach in graduate-level GPU systems courses and apply in production research labs.

---

## 1. Performance Philosophy for Agentic CUDA Development

1. **Treat the GPU as a memory hierarchy first, compute engine second.** Measure arithmetic intensity and ensure every optimization increases bytes reused per byte loaded from DRAM. Hopper’s 3 TB/s HBM bandwidth is the limiting reagent unless kernels surpass ~32 FLOP/B.
2. **Design for warps, orchestrate at blocks, validate at the grid.** Every transformation—tiling, prefetching, fusion—should preserve warp coalescing and avoid divergence before scaling outward.
3. **Iterate with instrumentation.** Profiling, sanitization, and search loops turn kernel engineering into a scientific process rather than ad hoc tuning. An AI agent must close the loop on every iteration.
4. **Fuse only with a dataflow budget.** Kernels that hoard registers or shared memory collapse occupancy; evaluate resource usage quantitatively before combining stages.
5. **Optimize for the target SM generation.** Expose architectural parameters (warp size, shared memory per SM, tensor core instruction set) in the agent’s context to avoid “lowest common denominator” kernels.

---

## 2. Memory-Centric Kernel Architecture

### 2.1 Global Memory Mastery
- **Coalescing Checklists.** Agents should auto-verify that every warp issues ≤4 memory transactions for 4-byte data. Embed stride analysis tools that flag misaligned base pointers or irregular strides.
- **Iterator Abstractions.** Encourage data-structure adaptors that present unit-stride views (e.g., CUTLASS `TensorRef`). This keeps generated kernels agnostic to underlying layouts but ensures coalesced access.
- **Async Copy Patterns.** On Ampere+, prefer `cp.async` or Hopper’s TMA to stage tiles into shared memory without polluting registers. Agents must coordinate `cp.async.commit_group`/`wait_group` or TMA barriers to avoid read-after-write hazards.

### 2.2 Shared Memory as a Software-Managed Cache
- **Bank-Conflict Vaccination.** Default to padded leading dimensions (`+1` column) or XOR swizzles. Agents can infer padding requirements via `(tile_dim % 32 == 0)` heuristics.
- **Multi-Stage Pipelines.** Alternate producer/consumer stages so that while one tile computes, another tile loads. Hopper TMA plus `mbarrier` enables 2–4 stage pipelines with minimal synchronization.
- **Cluster-Level SMEM.** On Hopper, Thread Block Clusters with Distributed Shared Memory allow 256 KB–1 MB of SMEM. AI agents should escalate to clusters only when arithmetic intensity justifies inter-block synchronization overhead.

### 2.3 Register Pressure and Occupancy
- **ILP vs. Occupancy Ledger.** Have the agent compute register counts, theoretical occupancy, and expected latency-hiding. Aim for 60–75% occupancy while sustaining ≥4-way ILP.
- **Launch Bounds as Contracts.** Use `__launch_bounds__` pragmatically to cap registers. Agents should emit comments explaining the bound selection and expected occupancy.
- **Register File Locality.** Promote register tiling (e.g., 4×4 micro-tiles) that fits within the warp’s register allocation while ensuring vectorized math instructions.

---

## 3. Warp-Level Excellence

### 3.1 Communication
- Prefer warp shuffle intrinsics (`__shfl_*`) for reductions, scans, and halo exchange inside a warp. Template helpers can wrap shuffle patterns for clarity and reuse.
- Cooperative Groups enable warp-synchronous algorithms without full-block barriers—agents should rely on `coalesced_group` and `tiled_partition` to formalize warp partitions.

### 3.2 Control Flow
- **Divergence Budgets.** Enforce warp-aligned conditionals (`if (lane_id < tile_width)`) and rely on predicate math for small branches.
- **Warp Specialization.** Hopper’s WGMMA benefits from assigning dedicated warps to loads vs. math. Agents should emit specialization diagrams documenting warp roles for maintainability.

### 3.3 ILP Scheduling
- **Register Blocking.** Generate multiple accumulators to keep both math pipelines occupied. On Hopper, target 8 FMA groups per warp to saturate dual-issue schedulers.
- **Latency Hiding via Triple Buffering.** Combine asynchronous copies, compute, and epilog operations into three overlapping stages.

---

## 4. Tensor Core Utilization

1. **Choose the Right Instruction Family.**
   - Ampere: `mma.sync`/WMMA fragments for 16×16×16 FP16/BF16/TF32.
   - Hopper: `wgmma.mma_async` with 128-thread warp groups, operand B from shared memory, asynchronous pipelines.
2. **Operand Layout Discipline.** Force row-major A, col-major B fragments to match tensor core expectations; convert via CUTLASS layout utilities or inline transposition when necessary.
3. **Epilogue Fusion.** Fold bias, activation, and normalization into tensor-core epilogues to avoid extra kernel launches.
4. **Precision Ladder.** Agents should understand FP8/FP16 accumulation strategies, scaling factors, and dynamic range calibration when integrating with frameworks like Transformer Engine.

---

## 5. Kernel Fusion and Dataflow Design

- **Bandwidth Accounting.** Before fusing, compute bytes moved by separate kernels vs. fused version. Fuse only when the reduction outweighs increased register/shared memory usage.
- **Streaming Softmax & Attention Patterns.** Adopt the FlashAttention design—tile Q/K/V, maintain running max & sum for numerical stability, and flush partial outputs incrementally.
- **Graph-Level Scheduling.** Combine CUDA Graphs with fused kernels to minimize launch latency for iterative inference/training loops.

---

## 6. Instrumented Iterative Optimization Loop

| Stage | Agent Role | Tooling | Key Metrics |
|-------|------------|---------|-------------|
| 1. Generate | Planner + Synthesizer | Model prompt with hardware context | Resource estimates, launch parameters |
| 2. Compile | Builder | `nvcc`, `ptxas`, Clang CUDA | Error logs, register counts |
| 3. Sanitize | Correctness Guardian | Compute Sanitizer (memcheck/racecheck/initcheck/synccheck) | Zero critical findings |
| 4. Validate | Unit Tester | CPU parity tests, tolerance-aware comparisons | max(|Δ|)/(ε_abs + ε_rel·|ref|) |
| 5. Profile | Performance Analyst | Nsight Systems/Compute, CUPTI counters | SOL %, dram__bytes, smsp__sass_average_active |
| 6. Optimize | Evolutionary Strategist | Auto-tuners (Triton, Kernel Tuner), Bayesian search | Speedup vs. baseline, config Pareto front |
| 7. Archive | Knowledge Curator | Innovation archive with metadata | Code fingerprint, profiler summary |

Agents should persist per-iteration telemetry (register count, occupancy, dram transactions, achieved FLOPs) to condition future prompts and build regression baselines.

---

## 7. Auto-Tuning and Search Strategies

- **Configuration Spaces.** Cover tile sizes, warp counts, stage depths, unroll factors, and precision modes. Encode constraints (e.g., `tile_m * tile_n * dtype_size ≤ shared_mem_limit`).
- **Hybrid Search.** Start with Latin Hypercube sampling, refine with Bayesian optimization, and finish with evolutionary crossover using the innovation archive.
- **Transfer Learning.** Seed new tasks with configurations from similar operators (e.g., matmul → attention) using embedding similarity over kernel metadata.

---

## 8. Integration with ML Frameworks

### PyTorch 2.4+
- Register operators with `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL`. Provide fake implementations for `torch.compile` tracing.
- Use `at::TensorAccessor` or `PackedTensorAccessor64` for boundary-safe indexing.
- Manage streams with `at::cuda::getCurrentCUDAStream()` and expose stream-ordered semantics for overlapping kernels.

### JAX / Pallas / Triton
- Prototype in Triton or Pallas for rapid iteration; drop to CUDA when profiling shows >10% headroom.
- For XLA custom calls, export a C API shim that wraps kernel launches, ensuring stream-awareness and shape polymorphism.

### Mixed-Precision & Distributed Contexts
- Integrate with Transformer Engine or Apex for loss scaling and FP8 metadata management.
- Coordinate kernel launches with NCCL collectives for tensor parallel workloads; ensure communicators overlap with compute via separate streams.

---

## 9. Validation, Fuzzing, and Reliability Engineering

1. **Deterministic Test Harnesses.** Seed random generators and capture reference outputs for reproducibility.
2. **Coverage-Aware Fuzzing.** Use libFuzzer-style harnesses emitting shape/value pairs, instrumented with `__sanitizer_cov_trace_pc`. Prioritize pathological dimensions (1, 31, 32, 33, prime numbers).
3. **Numerical Diagnostics.** Log ulp error histograms, detect catastrophic cancellation, and introduce Kahan-style compensated sums when error budgets exceed tolerances.
4. **Resilience Testing.** Validate kernels under reduced clocks (using `nvidia-smi -lgc`) to surface race conditions that only appear when instruction latencies fluctuate.

---

## 10. Knowledge Systems for Agentic CUDA Teams

- **RAG Corpus Structure.** Partition documents into: (a) Architecture Notes, (b) Optimization Patterns, (c) Kernel Templates, (d) Error Cookbook. Tag entries with compute capability, memory assumptions, and precision.
- **Prompt Scaffolding.** Supply the agent with ordered reasoning steps: _analyze data movement → select tiling → schedule memory ops → schedule compute → estimate utilization_. Encourage the agent to emit intermediate rationale for auditability.
- **Innovation Archive Schema.** Store `{kernel_hash, operation, shape_range, achieved_TFLOPs, dram_gbps, resource_usage, code}` to enable reuse and meta-learning.

---

## 11. Exemplars and Continuing Education

- **FlashAttention 1→3 Progression.** Study how each version layered optimizations: streaming softmax → tile pipelining → WGMMA + TMA. Replicate this pattern in new domains (e.g., attention-like graph ops).
- **CUTLASS & CuTe.** Treat these as canonical references for tensor core kernels. Agents should learn to emit CuTe tensor layout transforms to cut down on manual pointer arithmetic.
- **Academic & Industry Resources.** Incorporate GTC Hopper deep dives, Colfax Research tutorials, and Simon Bøhm’s matmul series into the RAG knowledge base. Update quarterly as GPU architectures evolve.

---

## 12. Checklist for Production-Ready Agent-Generated Kernels

- [ ] Compiles with `-arch=sm_80` and `-arch=sm_90` without warnings.
- [ ] Passes Compute Sanitizer suite (memcheck, racecheck, initcheck, synccheck).
- [ ] Achieves ≥70% memory bandwidth utilization _or_ ≥60% tensor core utilization relative to roofline.
- [ ] Demonstrates <1e-5 relative error (FP32) or <1e-2 (FP16/BF16) across 20 randomized test cases.
- [ ] Includes profiler attestation: achieved TFLOPs, DRAM GB/s, occupancy, stall breakdown.
- [ ] Documented resource usage: registers/thread, shared memory, launch bounds.
- [ ] Archived in knowledge base with metadata, profiler capture, and correctness certificate.

Following this blueprint enables AI agents to iteratively reach expert-crafted performance while preserving the scientific rigor demanded in production research settings.
