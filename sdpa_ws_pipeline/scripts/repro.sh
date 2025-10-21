#!/usr/bin/env bash
set -euo pipefail

# One‑click end‑to‑end: bench → tune → profile → report
# Requires: CUDA GPU, PyTorch (2.1+ recommended), Triton (if using Triton candidates), Nsight Compute CLI

echo "== Repro start =="
python3 - <<'PY'
import platform, torch, json, subprocess, os, sys, pathlib, time
env = {
  "python": platform.python_version(),
  "torch": getattr(torch, "__version__", "N/A"),
  "cuda": getattr(torch.version, "cuda", None),
  "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
  "sm": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}
pathlib.Path("artifacts").mkdir(exist_ok=True)
with open("artifacts/ENV.json","w") as f: json.dump(env,f,indent=2)
print(json.dumps(env, indent=2))
PY

# 1) Bench baselines + default candidates
bash scripts/bench.sh

# 2) EvoEngineer‑Full autotune
python3 scripts/evo_tune.py || true

# 3) Profile with Nsight Compute
bash scripts/profile.sh || true

# 4) Build summary report
python3 scripts/summarize.py

echo "== Repro done =="
echo "Artifacts: artifacts/*; Report: reports/summary.md"
