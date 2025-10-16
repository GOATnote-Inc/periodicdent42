# CUDA Kernels: Surgical Edits Only

**AI Rule**: Prefer targeted, surgical edits to kernels.

- Avoid global rewrites unless explicitly requested
- Test correctness after every change (`make bench-correctness`)
- Document performance deltas in commit messages
- Use region markers (see `fa_s512_v3.cu` for examples)

