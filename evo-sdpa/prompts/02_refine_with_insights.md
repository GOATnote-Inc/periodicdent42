You are refining our fused SDPA kernel **using Nsight Compute evidence**. Apply **only** changes with analytical benefit on Ada (sm_89). Keep correctness.

**Profiling evidence (paste key Nsight deltas here)**:
- `sm__pipe_tensor_active`: <X>%
- `smsp__inst_executed_pipe_lsu`: <value>; `smsp__gmio_stall`: <pct>
- `dram__throughput`: <pct>; `l1tex__data_bank_conflicts`: <value>

### Proven insights to apply (I3)
- **Pipeline depth**: 3‑stage helps L≥2048; 2‑stage better at ≤512 (SMEM pressure).
- **CTA shape**: For d=64, (M,N)=(128,64) > (64,128) (Q reuse). For d=128, invert to ease register pressure.
- **Epilogue micro‑fusion**: fold normalize + cast + (optional) `st.global.cs`.
- **Occupancy**: 2 CTAs/SM worked best; avoid >72 regs/thread.
- **(Deferred)** swizzle: use **interleaved column‑major** compatible with WMMA loads (not XOR that breaks contiguous columns).

### Your task
- Produce a **delta patch** to our current kernel file (v7a) that integrates the applicable I3 items, each tagged:
  `// INSIGHT:<reason>`
- Keep WMMA + store→softmax→rebuild and per‑warp ownership unchanged.
- Respect SMEM ≤ 64 KiB/CTA (unless justified), regs ≤ 72/thread.

**Return only the patched file.**

