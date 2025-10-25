# TMA Emission Investigation - Technical Findings

## Overview

Systematic investigation of TMA operation emission in Triton 3.0.0 on NVIDIA H100 using positive/negative control methodology with IR-first verification.

## Experimental Design

### Test Matrix

Three kernel configurations tested:

1. **Positive Control** (`copy_bp`): All TMA invariants satisfied
2. **Negative Control** (`copy_bp_bad`): Intentionally broken invariant
3. **Stress Test** (`stress_bp`): Complex patterns from Issue #6638

### TMA-Friendly Configuration

```
Tile dimensions: 128 × 128 (multiple of 16)
Memory layout: Row-major, unit-stride inner dimension
Block pointer order: (1, 0) - inner-first
Pipeline: num_stages=4, num_warps=8
Data type: float16
Boundary handling: Masked loads/stores
```

## Observations

### TTIR Analysis

All tested kernels emit standard Triton IR operations:

```mlir
%24 = tt.load %14, %23 : tensor<128x128x!tt.ptr<f16>>
tt.store %26, %24, %23 : tensor<128x128x!tt.ptr<f16>>
```

No `ttng.async_tma_load` or `ttng.async_tma_store` operations observed.

### TTGIR Analysis

Zero async operations in Triton GPU IR across all configurations.

### PTX Analysis

Generated PTX contains standard global memory operations:

```ptx
st.global.v4.b32 [ %rd51 + 0 ], { %r82, %r83, %r84, %r85 };
```

No `cp.async.bulk.tensor` instructions present.

## Control Validation

Negative control (`copy_bp_bad`) with deliberately incorrect `order=(0,1)` also shows zero TMA operations, confirming the absence is not due to compiler rejection of invalid configurations.

## Checksums

All IR artifacts SHA256-hashed for verification:

```
P0 TTIR: c6e4ff84fd8298fcecf91dfd9a1cbb36e33ed0b1b47af97d9fd6d326b4ed8372
Bundle:  8c5ee08e71a408116d707ce3a56664dd7c423610257a193d1ed96d0a87120fef
```

## Implications

1. Block pointer API in Triton 3.0.0 does not trigger TMA lowering
2. All tested configurations lower to standard `tt.load`/`tt.store`
3. Cannot validate Issue #6638 without actual TMA emission
4. Investigation should continue with:
   - Triton nightly/development builds
   - Explicit tensor descriptor API (if available)
   - Consultation with Triton maintainers

## Tested Invariants

All combinations tested:

- Tile sizes: 64×64, 128×128, 256×256
- num_stages: 3, 4, 5
- num_warps: 4, 8, 16
- Order: (0,1), (1,0)
- Descriptor patterns: Static, dynamic (in-loop)
- Store patterns: Unconditional, predicated

Zero TMA emission across all variations.

## Hardware Verification

```
$ nvidia-smi
GPU: NVIDIA H100 80GB HBM3
SM: 90 (Hopper architecture)
Driver: 575.57.08
CUDA: 12.9
```

Hardware supports TMA operations. Confirmed via device properties.

## Reproducibility

Complete test harness provided:
- Unique dump directories prevent IR reuse
- SHA256 checksums enable verification
- Environment snapshots included
- All source code provided

## Next Steps

1. Query Triton maintainers on TMA emission requirements
2. Test development/nightly Triton builds
3. Investigate internal TMA usage in `tl.dot`/`tl.attention`
4. Consider CUTLASS direct implementation as reference

## References

- Triton Issue #6638: TMA Store Non-deterministic results
- NVIDIA Hopper Architecture Whitepaper
- Triton Documentation: Block Pointers
- CUDA Toolkit: `cp.async.bulk.tensor` PTX instruction

