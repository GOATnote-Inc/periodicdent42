# All Workflows Disabled

**Date:** November 1, 2025

## Status

All GitHub Actions workflows are **DISABLED** for this repository.

## Why

- No CI/CD needed for research repository
- Reduces noise and unnecessary failures
- Focus on code, not automation

## Disabled Workflows

1. **compliance.yml** - Attribution compliance (moved to local git hooks)
2. **pages.yml** - GitHub Pages deployment (Pages disabled in repo settings)

## Local Checks

**Pre-commit hook** at `.git/hooks/pre-commit` handles attribution compliance locally.

**To disable GitHub Pages entirely:**
1. Go to Settings → Pages
2. Select "None" for branch
3. Save

This stops the automatic "pages build and deployment" workflow.

---

**All automation runs locally. No GitHub Actions needed.**

