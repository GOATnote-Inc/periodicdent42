# Status: FlashAttention S=512 (L4 Ada) — Evidence & Bench Readiness

## Snapshot (auto-updated by humans)
- Branch: `feature/evidence_wmma_tc`
- Build: **Release** (no DEBUG_V3, `-DNDEBUG`, `-DUSE_WMMA`)
- GPU: NVIDIA L4 (sm_89)

## Evidence (committed)
- Sanitizer: `cudadent42/artifacts/sanitizers/compute-sanitizer.log` → **0 errors**
- PTXAS: `cudadent42/artifacts/stats/ptxas.txt` → **30–32 regs**, **0 spills**, **SMEM 24–45KB**, **stack=0B**
- WMMA Proof: `cudadent42/artifacts/stats/wmma_proof.txt` or `artifacts/bench/bench.log` (mma.hpp warnings)

## Known Issue (bench loop)
```
RuntimeError: CUDA error: unspecified launch failure
```
- Occurs only in **repeated** launch loop (warmup/bench)
- Single call succeeds; correctness validated
- Not WMMA-specific (reproduced with no-WMMA)
- Suspected cause: S_row init loop or sync pattern during warmup

## Repro Commands
```bash
rm -rf ~/.cache/torch_extensions/* /tmp/torch_extensions/* || true
python3 -c "from build_v3_release import build_v3_release; build_v3_release(False)"
pytest -q tests/test_sdpa_parity.py || true
python3 scripts/bench_s512_tc_vs_sdpa.py || true
python3 scripts/summarize_s512_bench.py || true
```

## Next Actions (ordered)
1. **Keep GPU alive** (no auto-stop in any workflow/scripts)
2. Stabilize release bench:
   - Remove per-iteration `cudaStreamSynchronize()` (bindings)
   - Revert S_row init loop (see `DISABLE_SROW_INIT` guard)
   - Test 25–50 calls once, then summarize
3. If still failing:
   - `CUDA_LAUNCH_BLOCKING=1` and `TORCH_USE_CUDA_DSA=1` for one loop
   - `compute-sanitizer --tool racecheck` on loop
4. Collect S=512 JSON + `S512_BENCH_SUMMARY.md`
5. EvoEngineer targeted sweep (tiles, STAGES, maxrregcount)

## For LLMs
- Do not modify CI to stop GPU. Search workflows for `gcloud compute instances stop` and fail the job if found.
- Edit only the files listed in diffs; keep code-first style.
- When unsure, add evidence to `cudadent42/artifacts/**`.

