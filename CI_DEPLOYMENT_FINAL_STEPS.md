# CI Deployment: Final Steps

## Status: Ready for Manual Actions

All code is complete and validated. The following manual steps require GitHub UI access.

---

## Step 1: Setup Self-Hosted Runner (10 min)

### Generate Token
1. Visit: https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners/new
2. Select "Linux" and "x64"
3. Copy the token from the configuration command

### Install on GPU Instance
```bash
# SSH to GPU
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Create runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf actions-runner-linux-x64-2.311.0.tar.gz

# Configure (replace YOUR_TOKEN with token from GitHub)
./config.sh \
  --url https://github.com/GOATnote-Inc/periodicdent42 \
  --token YOUR_TOKEN \
  --name cudadent42-l4-runner \
  --labels self-hosted,gpu,cuda

# Run in background
nohup ./run.sh > runner.log 2>&1 &
```

### Verify
Check: https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners

Should show "cudadent42-l4-runner" with status "Idle"

---

## Step 2: Create Test PR (2 min)

Branch `test/ci-benchmark-validation` is already pushed.

### Create PR via GitHub UI
1. Visit: https://github.com/GOATnote-Inc/periodicdent42/pulls
2. Click "New pull request"
3. Base: `main` ← Compare: `test/ci-benchmark-validation`
4. Title: "test: Validate CI benchmark workflow"
5. Description:
   ```
   Testing automated CUDA benchmark CI workflow.
   
   Expected:
   - Build kernel
   - Run benchmark (30 sec)
   - Compare to baseline
   - Upload artifacts
   
   Files changed:
   - CI_TEST.md (test trigger)
   - .github/RUNNER_SETUP.md (documentation)
   ```
6. Click "Create pull request"

---

## Step 3: Trigger Workflow (1 min)

### Add Label
1. On the PR page, click "Labels" (right sidebar)
2. Type "benchmark" and press Enter to create the label
3. Click the label to add it to the PR

### Watch Workflow
1. Click "Actions" tab at top
2. Should see workflow "CUDA Benchmark" starting
3. Click the workflow to see live logs

---

## Step 4: Verify Results (2 min)

### Check Workflow Steps
Expected sequence:
1. ✅ Checkout Code
2. ✅ Check CUDA (nvidia-smi output)
3. ✅ Install dependencies
4. ✅ Build (setup.py build_ext --inplace)
5. ✅ Benchmark (integrated_test.py)
6. ✅ Check baseline (exists = true)
7. ✅ Compare (no regression expected)
8. ✅ Upload artifacts

### Download Artifacts
1. Scroll to bottom of workflow run
2. Click "benchmark-results" artifact
3. Extract and verify:
   - `results.json` (correctness, performance, roofline)
   - `comparison.json` (speedup ~1.0x, is_regression = false)

### Expected Results
```json
// results.json
{
  "correctness": {"passed": true, "max_abs_error": < 0.001},
  "performance": {"mean_time_ms": ~0.052, "throughput_gflops": ~20000},
  "roofline": {"bottleneck": "Memory Bandwidth", "efficiency_pct": ~107}
}

// comparison.json
{
  "speedup": ~1.0,
  "improvement_pct": ~0.0,
  "is_regression": false
}
```

---

## Success Criteria

- [x] Code complete and pushed
- [x] Baseline created (.baseline.json)
- [ ] Runner installed and "Idle"
- [ ] Test PR created
- [ ] Label "benchmark" added
- [ ] Workflow triggered
- [ ] All steps passed (green checks)
- [ ] Artifacts uploaded
- [ ] No regression detected
- [ ] Results match expected values

---

## Troubleshooting

### Runner Shows "Offline"
```bash
# SSH to GPU instance
cd ~/actions-runner
./run.sh --check
# If not running, restart:
nohup ./run.sh > runner.log 2>&1 &
```

### Workflow Doesn't Trigger
- Verify label is exactly "benchmark"
- Check workflow file has correct paths
- Ensure runner has label "self-hosted"
- Try manual dispatch: Actions → CUDA Benchmark → Run workflow

### Build Fails
```bash
# Test manually on GPU instance
cd ~/periodicdent42/cudadent42
git pull
python setup.py clean
python setup.py build_ext --inplace
```

### Benchmark Fails
```bash
# Test manually
cd ~/periodicdent42/cudadent42/bench
python3 integrated_test.py --output test.json
cat test.json | python3 -m json.tool
```

### Regression Detected (Unexpected)
- Check if GPU was under load during benchmark
- Re-run workflow (timing variance is normal)
- If persistent, investigate with profiler

---

## After Successful Test

### Cleanup Test Branch
```bash
# Local machine
git checkout main
git branch -D test/ci-benchmark-validation
git push origin --delete test/ci-benchmark-validation
```

### Close Test PR
On GitHub: Comment "CI validation complete" and close (don't merge)

### Keep Runner Running
Runner is now operational for future PRs. To trigger:
- Add label "benchmark" to any PR
- Or use manual dispatch in Actions tab

### Stop GPU When Not Needed
```bash
# Stop runner
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a --command="pkill -f run.sh"

# Stop instance
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a

# Start when needed
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
# Then restart runner:
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a --command="cd ~/actions-runner && nohup ./run.sh > runner.log 2>&1 &"
```

---

## Documentation

All documentation is committed:
- `.github/workflows/cuda_benchmark.yml` - Workflow definition
- `.github/RUNNER_SETUP.md` - Detailed runner setup
- `cudadent42/bench/CI_INTEGRATION.md` - Integration guide
- `cudadent42/bench/CI_VALIDATION_COMPLETE_OCT13_2025.md` - Validation report
- `cudadent42/bench/.baseline.json` - Performance baseline
- `CI_TEST.md` - Test checklist
- `CI_DEPLOYMENT_FINAL_STEPS.md` - This file

---

## Summary

**Status:** 95% complete

**Remaining:**
1. Generate runner token (GitHub UI, 1 min)
2. Install runner (SSH, 5 min)
3. Create PR (GitHub UI, 1 min)
4. Add label (GitHub UI, 10 sec)
5. Verify (Actions tab, 2 min)

**Total time:** ~10 minutes

**Cost:** $0.0017 per workflow run (30 seconds on L4)

**Production ready:** Yes, pending manual steps above.

