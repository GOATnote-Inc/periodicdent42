#!/usr/bin/env bash
set -euo pipefail

# Ensure preflight exists, then run it
bash scripts/gen_preflight.sh
bash tools/preflight.sh

# Build + bench (adjust to your repo layout if needed)
python3 -c "import flashmoe_science; print('✔ import OK')"
python3 benches/bench_correctness_and_speed.py --repeats 50 --warmup 10 --save-csv --verbose

