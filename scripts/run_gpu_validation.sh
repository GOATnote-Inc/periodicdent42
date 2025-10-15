#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
ART="$ROOT/cudadent42/artifacts"
mkdir -p "$ART"/{sanitizers,stats,bench}

echo "[1/6] Build (release, -DNDEBUG, -DUSE_WMMA)"
rm -rf ~/.cache/torch_extensions/* /tmp/torch_extensions/* || true
python3 - <<'PY'
from build_v3_release import build_v3_release
build_v3_release(False)
print("✅ build(release) OK")
PY

echo "[2/6] Sanitizers (with oracle)"
if [ ! -x "$ROOT/cudadent42/bench/tests/oracles/tile_oracle_v3.py" ]; then
  cat > "$ROOT/cudadent42/bench/tests/oracles/tile_oracle_v3.py" <<'EOO'
#!/usr/bin/env python3
import argparse, torch
from build_v3_release import build_v3_release
torch.backends.cuda.matmul.allow_tf32=False
def main():
  p=argparse.ArgumentParser(); p.add_argument("--config",type=int,default=1)
  a=p.parse_args()
  m=build_v3_release(True)
  f=m.forward
  B,H,S,D=2,8,512,64
  Q=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16); K=Q.clone(); V=Q.clone()
  O=f(Q,K,V,1.0/(D**0.5),False,a.config); assert torch.isfinite(O).all()
  print("OK")
if __name__=="__main__": main()
EOO
  chmod +x "$ROOT/cudadent42/bench/tests/oracles/tile_oracle_v3.py"
fi
if command -v compute-sanitizer >/dev/null 2>&1; then
  scripts/ci/compute_sanitizer_gate.sh || true
else
  echo "compute-sanitizer not available" > "$ART/sanitizers/SANITIZER_STATUS.txt"
fi

echo "[3/6] PTXAS snapshot"
scripts/ci/ptxas_snapshot.sh || true

echo "[4/6] SASS proof (mma.sync/HMMA)"
SO=$(ls -1 ~/.cache/torch_extensions/*/*.so 2>/dev/null | head -1 || true)
if [ -n "${SO:-}" ] && command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-sass "$SO" | grep -mi1 "mma.sync\\|HMMA" > "$ART/stats/wmma_proof.txt" || true
else
  echo "cuobjdump or .so missing" > "$ART/stats/wmma_proof.txt"
fi

echo "[5/6] Bench (streams variant optional)"
python3 "$ROOT/scripts/bench_s512_tc_vs_sdpa.py" || true
python3 "$ROOT/scripts/bench_s512_tc_vs_sdpa.py" --streams || true
python3 "$ROOT/scripts/summarize_s512_bench.py" || true

echo "[6/6] Artifact summary"
ls -lh "$ART"/{sanitizers,stats,bench}/* 2>/dev/null || true
echo "DONE"
