# Triton TMA Emission Investigation (H100)

Investigation of Tensor Memory Accelerator (TMA) operation emission in OpenAI Triton 3.0.0 on NVIDIA H100 GPUs.

## Objective

Determine whether Triton 3.0.0 emits `ttng.async_tma_*` operations for user-defined kernels on SM90 (H100) hardware when all documented TMA-friendly invariants are satisfied.

## Methodology

### Control Test Design

Three-phase validation using IR-first verification:

1. **P0 (Positive Control)**: Clean TMA-friendly kernel with optimal configuration
2. **P1 (Negative Control)**: Deliberately broken invariant to validate detection
3. **P2 (Stress Test)**: Issue #6638 patterns (descriptor-in-loop, predication, tails)

### Verification Approach

- **IR Oracle**: Scan TTIR/TTGIR for `async_tma` operations (not PTX-only)
- **Checksums**: SHA256 hashing of all IR artifacts
- **Isolation**: Unique dump directories per run
- **Stop Condition**: Halt at P0 failure per scientific methodology

## Results

### Summary

| Test | Kernel | async_tma | Result |
|------|--------|-----------|--------|
| P0 | `copy_bp` | 0 | No TMA emission |
| P1 | `copy_bp_bad` | 0 | Control valid |
| P2 | `stress_bp` | 0 | Cannot test |

### P0 Configuration

```python
Shape: 4096 × 4096
Tiles: 128 × 128
Strides: (N, 1)           # Unit-stride inner dimension
Order: (1, 0)             # Inner-first layout
num_stages: 4
num_warps: 8
dtype: float16
```

### IR Evidence

**TTIR**: Shows `tt.load` and `tt.store` operations  
**TTGIR**: Zero async operations  
**PTX**: Contains `st.global` (not `cp.async.bulk.tensor`)

## Environment

```
GPU: NVIDIA H100 80GB HBM3
Compute Capability: SM90
Driver: 575.57.08
CUDA: 12.9
Triton: 3.0.0
PyTorch: 2.4.1+cu124
```

## Reproducing

```bash
# Run control test
cd tma-issue-6638-reproducer
bash tma_final_control_test.sh

# Check results
cat tma-verify-*/artifacts/ir/summary.tsv
```

## Artifacts

All IR dumps are checksummed (SHA256) and bundled:
- Environment snapshots
- Kernel source
- TTIR/TTGIR/PTX dumps
- Execution logs

## Files

```
tma-issue-6638-reproducer/
├── tma_final_control_test.sh       # Main control test
├── scripts/
│   ├── run_and_capture.sh          # IR capture harness
│   ├── tma_user_repro.py           # User kernel example
│   ├── determinism_and_sanitizers.sh
│   └── tma_invariant_sweep.py
└── artifacts/                      # Generated IR dumps
```

## Question for Triton Maintainers

Which invariant is required for `ttng.async_tma_*` lowering on SM90 in Triton 3.0.0?

Specifically:
1. Is TMA emission supported for user kernels in this version?
2. If yes, what configuration triggers IR lowering to async TMA?
3. If no, which Triton version or branch provides user-facing TMA?
4. Is TMA reserved for internal operations only?

## Related

- Triton Issue #6638: TMA Store Non-deterministic results
- Hardware: NVIDIA Hopper Architecture (SM90)
- API: Block pointers (`tl.make_block_ptr`)

## License

See repository root LICENSE file.
