NON-NEGOTIABLE:
1) Always run `bash scripts/gen_preflight.sh && bash tools/preflight.sh` FIRST.
2) If either step fails or `tools/` is missing: STOP. Print the exact failing command/output. Do not infer or continue.
3) Only after preflight passes: build, then benchmarks.

Rationale: prevents wasted GPU minutes and false narratives about the environment.

