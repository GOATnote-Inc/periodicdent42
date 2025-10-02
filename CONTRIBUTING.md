# Contributing Guide

## Branching & Workflow
- Fork or branch from `main`. Use feature branches named `feature/<slug>` or `chore/<slug>`.
- Keep commits small and descriptive (imperative mood). Reference audit items or tickets when applicable.
- All changes ship via pull request with at least one reviewer approval.

## Required Checks
Before requesting review run:

```bash
make lint
make test
make audit
make graph
```

Include command output in the PR description.

## Code Style
- **Python**: PEP 8 via `ruff`, type hints on new/modified functions. Avoid global state—prefer dependency injection.
- **TypeScript/React**: Functional components with explicit prop types. Client components marked with `"use client"` when stateful.
- **Docs**: Use Markdown, wrap at 100 characters where practical, add file-level overviews when introducing new folders.

## Testing Expectations
- Add or update unit tests whenever logic changes.
- Integration tests should stub external services (Gemini, Vertex AI, hardware drivers).
- For demo UIs add screenshot or recording under `docs/screenshots/`.

## Reviewing Checklist
Reviewers should confirm:
- CI commands above succeed.
- `.env.example` updated when new variables are introduced.
- No secrets checked in (`git secrets --scan` if unsure).
- Docs and changelog entries reflect shipped behaviour.

## Release Notes
Update `CHANGELOG.md` in every PR with a short bullet summarising the change and linking to relevant docs or demos.
