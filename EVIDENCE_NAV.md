# Evidence Navigator

## Criticism → Artifact Map

### 1) Warp-Level Races
- Fix: lane-exclusive SMEM writes (no atomics): `cudadent42/bench/kernels/fa_s512_v3.cu` (search `lane_id`).
- Invariant: monotone online-softmax: same file (search `CUDA_DEBUG_ASSERT(l_new >= l_i`).
- Logs: `cudadent42/artifacts/sanitizers/` (memcheck/racecheck/synccheck or `SANITIZER_STATUS.txt`).

### 2) "No WMMA" (Tensor Cores)
- Code: `fa_s512_v3.cu` WMMA path (search `qk_dot_wmma_tile` and `mma_sync`).
- Build flags: `cudadent42/bench/build_v3_release.py` (`-DUSE_WMMA`).
- PTXAS: `cudadent42/artifacts/stats/ptxas.txt`.
- SASS proof: `cudadent42/artifacts/stats/wmma_proof.txt` (contains `mma.sync` / `HMMA.*`).

### 3) Benchmarks & Repro
- S=512 bench JSON: `cudadent42/artifacts/bench/tc_vs_sdpa_s512.json`.
- Repro scripts:
  - `scripts/ci/compute_sanitizer_gate.sh`
  - `scripts/ci/ptxas_snapshot.sh`
  - `scripts/bench_s512_tc_vs_sdpa.py`

## One-Command Repro (local)
```bash
scripts/bootstrap_tools.sh
pytest -q tests/test_sdpa_parity.py || true
pytest -q tests/test_tc_sdpa_parity.py || true
scripts/ci/compute_sanitizer_gate.sh || true
scripts/ci/ptxas_snapshot.sh
python scripts/bench_s512_tc_vs_sdpa.py || true
```

## Quick Proof Checks
```bash
grep -n "lane_id" cudadent42/bench/kernels/fa_s512_v3.cu
grep -n "mma_sync" cudadent42/bench/kernels/fa_s512_v3.cu || true
grep -i "HMMA\|mma.sync" cudadent42/artifacts/stats/wmma_proof.txt
cat cudadent42/artifacts/stats/ptxas.txt | sed -n '1,120p'
ls -lh cudadent42/artifacts/{sanitizers,stats,bench}
```
