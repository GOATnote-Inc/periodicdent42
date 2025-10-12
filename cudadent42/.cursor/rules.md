NON-NEGOTIABLE RULES:

## Before ANY GPU Instance Start:
1) **MANDATORY**: Run PRE_GPU_VALIDATION_CHECKLIST.md (8 steps, ~10 min, $0 cost)
2) **VERIFY**: All source files present (`python3 -c "from setup import ext_modules; ..."`)
3) **CONFIRM**: Preflight scripts exist (`ls tools/preflight.sh scripts/gen_preflight.sh`)
4) **CHECK**: Latest code pulled (`git pull origin cudadent42`)
5) **REVIEW**: Recent session docs (e.g., BENCHMARK_SESSION_*.md for last failures)

## On Remote GPU Instance:
1) Always run `bash scripts/gen_preflight.sh && bash tools/preflight.sh` FIRST
2) If either step fails or `tools/` is missing: STOP. Print exact error. Do not infer.
3) Use `python setup.py build_ext --inplace` (NOT manual nvcc commands)
4) Only after build succeeds: run import test, then benchmarks

## Cost Control:
- Set time limit (e.g., "stop after 20 minutes")
- Set cost budget (e.g., "max $1 this session")
- Track: start time, instance type, expected cost

## Rationale:
- October 12: Skipped local validation → $0.30 wasted on missing files
- October 11: No preflight → $4.61 wasted on 5 environment failures
- Existing docs: artifact_checklist.md, VALIDATION_BEST_PRACTICES.md patterns already solve this

**PREVENTION > RECOVERY**: 10 minutes local validation prevents $5-10 GPU waste

