# Cursor AI Rules (Repo-Wide)

## Editable Zones
Only modify files in:
- `cudadent42/bench/**` (benchmarks, build scripts)
- `cudadent42/bench/kernels/**` (CUDA kernels)
- `scripts/**` (dev/bench/profile helpers)
- `.github/**` (CI/CD)
- `infra/ci/**` (if needed for CI)
- `.cursor/**` (AI rules)
- `CODEMAP.md` (orientation map)
- `README.md` (top-level docs)

## Output Format
- **Primary**: Unified diffs + exact shell commands
- **Secondary**: Brief explanations only when asked
- Keep changes surgical and reversible

## Testing Requirements
- Add tests **before** refactors
- Run `make test` before committing
- Update `CODEMAP.md` if entrypoints change

## No-Go Actions
- Do NOT edit `docs/**` unless explicitly requested
- Do NOT edit `infra/**` (runtime infra) without approval
- Do NOT delete user content without archiving first
- Do NOT touch credentials or secrets

