#!/usr/bin/env bash
set -euo pipefail

# ---------- CONFIG ----------
ZONE="${ZONE:-us-west1-b}"
INSTANCE="${INSTANCE:-cudadent42-t4-dev}"
ARCH_LIST="${FA_ARCHS:-75}"    # T4 by default
PRESET="${FA_TILE_PRESET:-0}"  # 0=t4_safe, 1=ampere_balanced
# ----------------------------

trap 'echo "❌ Build failed at line $LINENO"; exit 1' ERR

echo "▶ Pre-flight GPU check"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo "unknown")
GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' || echo "0")
echo "GPU: $GPU_NAME (SM_$GPU_SM)"

if [[ "$GPU_SM" != "75" ]]; then
  echo "⚠️  Expected SM_75 (T4). Detected SM_$GPU_SM. Proceeding anyway."
fi

echo "▶ Clean + build (ARCHS=$ARCH_LIST, PRESET=$PRESET)"
export PYTHONPATH=".:${PYTHONPATH}"
python3 setup.py clean --all

# Deterministic build knobs also set inside setup.py; keep env clean.
FA_ARCHS="$ARCH_LIST" FA_TILE_PRESET="$PRESET" python3 setup.py build_ext --inplace

echo "▶ Import smoke test & symbol check"
python3 - <<'PY'
import importlib, sys
try:
    m = importlib.import_module("flashmoe_science")
    print("✅ Imported:", m.__name__, "version:", getattr(m, "__version__", "n/a"))
except Exception as e:
    print("❌ Import failed:", e)
    sys.exit(1)
try:
    import torch
    print("✅ Torch:", torch.__version__)
except Exception as e:
    print("⚠️  Torch not found:", e)
PY

# Ensure no bf16 symbols on SM_75 builds (best-effort)
if [[ "$GPU_SM" == "75" ]]; then
  so_path=$(find build -name "*.so" 2>/dev/null | head -n1 || true)
  if [[ -n "${so_path}" ]]; then
    if nm -D "$so_path" 2>/dev/null | grep -iq "bfloat16"; then
      echo "❌ BF16 symbols present on SM_75 build. Check gating."
      exit 1
    else
      echo "✅ No BF16 symbols (good for SM_75)."
    fi
  fi
fi

echo "▶ Run correctness tests"
python3 -m pytest tests/test_attention_correctness.py -v --maxfail=1 || {
  echo "⚠️  Tests failed. Check output above."
  exit 1
}

echo "▶ Perf sanity check"
if [ -f "benchmarks/benchmark_attention.py" ]; then
  python3 benchmarks/benchmark_attention.py --quick || echo "⚠️  Benchmark script not ready yet"
else
  echo "⚠️  Benchmark script not found, skipping perf check"
fi

echo "✅ Tests passed. Stopping GPU instance to avoid idle billing…"
gcloud compute instances stop "$INSTANCE" --zone="$ZONE" 2>/dev/null || {
  echo "⚠️  Could not auto-stop instance (may not have gcloud access)"
}

echo "🎉 Phase 2 sweep complete."

