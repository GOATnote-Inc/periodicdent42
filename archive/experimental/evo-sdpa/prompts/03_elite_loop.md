Role: CUDA Kernel Evolution Specialist under **elite preservation**.

Inputs:
- I1: `00_task.md`
- I2: Top‑K summary (paste the 3 fastest kernel notes: tile shapes, reg count, cp.async stages, SMEM bytes, kernel times)
- I3: Validated insights from the last profiling pass (`02_refine_with_insights.md`)

**Goal.** Propose **one child** that **keeps the winning tile family** and changes **exactly two** levers:
(1) pipeline_depth, (2) CTA (M,N,K), (3) warp specialization, (4) per‑warp MMA micro‑tile, (5) SMEM layout (padding/interleaved col‑major), (6) epilogue fusion.

**Rules.**
- Regs ≤ 72/thread (hard), SMEM ≤ 64 KiB/CTA unless justified.
- Preserve streaming softmax invariants and per‑warp row ownership.

**Output.**
- Updated kernel file with changes, tagged `// ELITE-CHG:<lever>`
- A five‑line **why it should be faster** note at the top

**Selection (outside this prompt).**
- Build → run bench → if ≥3% speedup and correctness holds, keep in Top‑K; else discard.

