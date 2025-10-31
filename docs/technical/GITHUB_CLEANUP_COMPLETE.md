# GitHub Actions & Dependabot Cleanup

**Date:** October 28, 2025  
**Status:** ✅ Fixed - Ready to commit

---

## Problem

- **5,092 workflow runs** spamming the Actions tab
- **7 open dependabot PRs** cluttering Pull Requests
- Workflows running on every push to `main`

---

## What Was Fixed

### 1. Disabled Problematic Workflows ✅

**Files Renamed (disabled):**
```
.github/workflows/compliance.yml        → compliance.yml.disabled
.github/workflows/pages.yml             → pages.yml.disabled
```

**Why:** Both workflows were triggering on every push to `main`, causing hundreds of runs.

### 2. Removed Dependabot Configuration ✅

**File Deleted:**
```
.github/dependabot.yml
```

**Why:** Dependabot was opening PRs for minor dependency updates, which conflicts with the FlashCore stability policy.

### 3. Created Cleanup Scripts ✅

**Created Scripts:**
- `cleanup_workflows.sh` - Deletes old workflow runs via GitHub API
- `cleanup_dependabot_prs.sh` - Closes open dependabot PRs

---

## How to Complete Cleanup

### Step 1: Commit Changes

```bash
cd /Users/kiteboard/.cursor/worktrees/periodicdent42/1761409560674-299b6b

# Check what changed
git status

# Stage changes
git add .github/workflows/compliance.yml.disabled
git add .github/workflows/pages.yml.disabled
git add cleanup_workflows.sh
git add cleanup_dependabot_prs.sh
git add GITHUB_CLEANUP_COMPLETE.md
git rm .github/dependabot.yml

# Commit
git commit -m "fix: Disable spammy workflows and dependabot

- Rename compliance.yml and pages.yml to .disabled
- Remove dependabot.yml (conflicts with stability policy)
- Add cleanup scripts for workflow runs and PRs
- Prevents 5,000+ workflow run accumulation"

# Push
git push origin main
```

### Step 2: Close Dependabot PRs

```bash
# Close the 7 open dependabot PRs
./cleanup_dependabot_prs.sh
```

**What it does:**
- Fetches all open PRs from `app/dependabot`
- Asks for confirmation
- Closes each PR with a comment explaining the policy

### Step 3: Delete Old Workflow Runs

```bash
# Delete the 5,092 workflow runs
./cleanup_workflows.sh
```

**What it does:**
- Fetches all workflows
- Deletes up to 100 runs per workflow
- May need to run multiple times for large backlogs

**Note:** GitHub API limits to 100 runs per request. For 5,092 runs, you may need to run this script ~50 times, or use the GitHub UI to bulk delete.

### Step 4: Verify Cleanup

1. Visit: https://github.com/GOATnote-Inc/periodicdent42/actions
   - Should show no new runs
   - Old runs should be deleted or significantly reduced

2. Visit: https://github.com/GOATnote-Inc/periodicdent42/pulls
   - Should show 0 open dependabot PRs

---

## Optional: Re-enable Workflows (Manual Only)

If you want to keep these workflows but only run them manually:

### Manual-Only Compliance Workflow

```yaml
# .github/workflows/compliance.yml
name: Attribution Compliance

on:
  workflow_dispatch:  # Manual only

jobs:
  check-attribution:
    # ... (rest of the job)
```

### Manual-Only Pages Workflow

```yaml
# .github/workflows/pages.yml
name: Deploy GitHub Pages

on:
  workflow_dispatch:  # Manual only

# ... (rest of the workflow)
```

To run manually:
1. Go to Actions tab
2. Select workflow
3. Click "Run workflow"

---

## Why These Changes

### Workflow Spam

**Problem:**
- Every commit to `main` triggered 2+ workflows
- Over time, accumulated 5,092+ runs
- Cluttered Actions tab, slowed down GitHub UI

**Solution:**
- Disable auto-trigger workflows
- Only run manually when needed
- FlashCore is a research repo, doesn't need CI on every commit

### Dependabot PRs

**Problem:**
- Dependabot opening PRs for minor version bumps
- Conflicts with FlashCore stability policy
- FlashCore pins exact versions (PyTorch 2.4.1, CUDA 13.0.2, etc.)

**Solution:**
- Remove dependabot.yml
- Manage dependencies manually per stability policy
- Only update for critical security issues (CVSS ≥ 7.0)

---

## Results After Cleanup

**Before:**
```
Actions:        5,092 workflow runs
Pull Requests:  7 dependabot PRs
Workflows:      2 auto-triggering on every push
```

**After:**
```
Actions:        0 workflow runs (or minimal after cleanup script)
Pull Requests:  0 dependabot PRs
Workflows:      0 auto-triggering (disabled)
```

---

## Maintenance

**Going Forward:**
- No auto-triggering workflows
- No dependabot PRs
- Manual dependency updates only
- Clean Actions tab
- Focus on research, not CI/CD overhead

**If Needed:**
- Re-enable workflows with `workflow_dispatch` only
- Run compliance checks manually before releases
- Keep GitHub UI clean and focused

---

## Summary

✅ Disabled spammy workflows (compliance, pages)  
✅ Removed dependabot configuration  
✅ Created cleanup scripts (workflows, PRs)  
✅ Ready to commit and push  

**Next Action:** Commit changes, then run cleanup scripts to remove old data.

---

**Status:** Production-Ready ✅

