# HARD RULES — DO NOT VIOLATE

- Environment is containerized. Only run and modify code that builds inside Docker.
- Toolchain is HARD-PINNED: CUDA Toolkit 13.0.2, CUTLASS 4.3.0, Nsight Compute CLI.
- Target arch is sm_90a (H100). Do NOT change to sm_90 or add cross-arch flags.
- No Triton, no FA3 kernels, no PyTorch SDPA in this repo. This project is CuTe/CUTLASS only.
- No "auto-fix": do NOT rewrite kernels, switch libraries, or install new packages.
- Keep BSR (block-sparse) layout, TMA loads, 2–3 stage pipeline, WMMA/FP32 accumulators.

# ALLOWED ACTIONS

- Edit code in src/ and scripts/ only if it compiles with nvcc -arch=sm_90a inside the container.
- Improve CuTe/CUTLASS wiring (TMA, mbarrier, swizzled SMEM) without changing libraries.
- Add Nsight Compute metric collection or CI to export .ncu-rep files.

# COMMANDS YOU MAY RUN (inside the container)

- make build
- make run
- make ncu
- make verify
- make clean

# FAILURE MODES TO AVOID

- Attempting to "upgrade/downgrade" CUDA or CUTLASS.
- Replacing CUTLASS with Triton, cuBLASLt heuristics, or PyTorch.
- Removing containerization or adding host-only scripts.

