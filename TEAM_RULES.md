# Team Rules – Autonomous R&D Intelligence Layer

These conventions are mandatory for all contributors and automation agents. They codify how we write code, review changes, and keep the platform secure.

## Coding Conventions
- **Languages:** Python 3.12+, TypeScript (frontend), Rust (safety kernel). Use toolchain files where provided.
- **Formatting:**
  - Python – `ruff format` + `ruff check` for linting.
  - TypeScript/JavaScript – `pnpm lint` (ESLint) and `pnpm format` (Prettier) when touching UI assets.
  - Rust – `cargo fmt` + `cargo clippy` with `-D warnings`.
- **Typing:**
  - Python modules must include type hints and pass `mypy --strict` when mypy configs exist.
  - TypeScript code must pass `tsc --noEmit`.
- **Testing:** Every change requires relevant unit tests. New modules ship with ≥ 80% coverage; critical paths (API endpoints, safety kernel) must remain ≥ 90%.

## Git & Review Process
- **Branch naming:** `feat/<scope>`, `fix/<scope>`, `chore/<scope>`, or `docs/<scope>`.
- **Commit style:** Conventional Commits (`type(scope?): summary`). Squash merges allowed only if individual commits follow the convention.
- **Pull requests:**
  - Include a risk assessment (low/medium/high) and rollback plan for medium+.
  - Reference any diagrams/screenshots used for context.
  - Request review from platform + domain owners for cross-cutting changes.

## Review Gates (Bugbot Enforcement)
Bugbot blocks merges unless all gates pass:
1. ✅ CI green (unit, integration, type checks).
2. ✅ Test coverage ≥ 80% overall and ≥ 90% on critical paths.
3. ✅ No secrets or tokens in the diff (hooks enforce redaction; Bugbot double-checks).
4. ⚠️ PRs touching > 500 LOC trigger a "large PR" warning requiring explicit maintainer approval.

## Supply Chain Policy
- Pin dependencies using lockfiles (`requirements.txt`, `poetry.lock`, `pnpm-lock.yaml`, `Cargo.lock`). Do not loosen ranges without security review.
- Run `npm audit fix --only=prod` or `pip install --require-hashes` equivalent before release branches.
- Deny any install scripts fetched via `curl`, `wget`, or similar. Use package managers with `--ignore-scripts` when adding new dependencies.
- For Python, prefer `uv` or `pip-tools` generated locks. Never use `pip install -U <pkg>` without an associated lock update.

## Platform Guidelines
- **OS:** Primary development on macOS or Linux shells. Use Bash-compatible scripts. Windows contributors should run inside WSL2.
- **GCP:** Use Workload Identity; `gcloud auth application-default login` is forbidden in CI and blocked by hooks.

## Dashboard Snippet
```
Team Rules:
- Enforce ruff/black, ESLint+Prettier, cargo fmt/clippy.
- Conventional Commits; branch = feat|fix|chore|docs/<scope>.
- Tests + type checks required; coverage ≥80% overall, ≥90% critical paths.
- Bugbot gates: CI green, coverage thresholds, secrets scan clean, warn on >500 LOC PRs.
- Pin deps; run npm audit --only=prod or pip hashes; block curl|bash installs.
- Use Bash or WSL2; forbid gcloud ADC login.
```

Adhering to these rules keeps AI-assisted development predictable, reviewable, and compliant.
