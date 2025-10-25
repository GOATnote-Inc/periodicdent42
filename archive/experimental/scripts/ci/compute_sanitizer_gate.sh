#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/../.. && pwd)"
ART="$ROOT/cudadent42/artifacts/sanitizers"
mkdir -p "$ART"
cd "$ROOT/cudadent42/bench"

# Debug build flags injected by build_v3_release(debug=True)
python3 - <<'PY'
from build_v3_release import build_v3_release
build_v3_release(debug=True)
print("✅ debug build")
PY

run() { echo -e "\n=== $1 ==="; shift; "$@" 2>&1 | tee "$ART/$1.log" ; }

run memcheck    compute-sanitizer --tool memcheck   python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal
run racecheck   compute-sanitizer --tool racecheck  python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal
run initcheck   compute-sanitizer --tool initcheck  python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal
run synccheck   compute-sanitizer --tool synccheck  python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal

echo "✅ sanitizer logs in $ART"

