# CI Benchmark Workflow Test

This file triggers the CI benchmark workflow when this branch is opened as a PR with the "benchmark" label.

## Test Configuration

- **Branch:** test/ci-benchmark-validation
- **Workflow:** `.github/workflows/cuda_benchmark.yml`
- **Trigger:** PR label "benchmark"
- **Expected:** Build, benchmark, compare, upload artifacts

## Test Checklist

- [ ] PR created
- [ ] Label "benchmark" added
- [ ] Runner online ("Idle" status)
- [ ] Workflow triggered in Actions tab
- [ ] Build step passed
- [ ] Benchmark step passed (30 seconds)
- [ ] Compare step passed
- [ ] Artifacts uploaded (results.json, comparison.json)
- [ ] No regression detected
- [ ] Workflow completed (green check)

## Expected Output

### results.json
```json
{
  "correctness": {"passed": true},
  "performance": {"mean_time_ms": ~0.052},
  "roofline": {"bottleneck": "Memory Bandwidth"},
  "config": {"batch_size": 32, "seq_len": 128}
}
```

### comparison.json
```json
{
  "speedup": ~1.0,
  "improvement_pct": ~0.0,
  "is_regression": false
}
```

## On Success

- Workflow shows green check
- Baseline unchanged (no regression)
- Ready to merge

## On Failure

Check workflow logs for:
- Build errors → Fix kernel code
- Correctness errors → Fix numerical issues
- Regression → Profile and optimize
- Runner errors → Check runner setup

## Cleanup

After successful test:
```bash
git checkout main
git branch -D test/ci-benchmark-validation
git push origin --delete test/ci-benchmark-validation
```

