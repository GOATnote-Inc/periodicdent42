#!/usr/bin/env python3
"""
EvoEngineer‑Full autotune/search
================================
Search over block sizes/tiling, num warps, stages, prefetch depth, shared‑mem layout,
epilogue fusion, attention masking path, dtypes, QK^T chunking, KV cache layout,
causal vs non‑causal, dropout off.

Artifacts:
- artifacts/tune/tune_log.csv
- artifacts/tune/topk.json
- kernels/candidate_*/ (snapshots of winning configs)

This script expects candidate backends to read tuning knobs from environment
or via kwargs (for Triton variants).
"""
import os, csv, json, time, itertools, shutil, subprocess, sys
from pathlib import Path

# Config space (focus on Triton candidates here; CUDA stub reads NVCC flags if used)
BACKENDS = [
    "candidate_triton_ws",
    "candidate_triton_flashlike",
    "candidate_cuda_stub",
]

GRID = {
    "BLOCK_M": [32, 64],
    "BLOCK_N": [32, 64],
    "BLOCK_D": [64],          # head dim fixed
    "NUM_WARPS": [2, 4, 8],
    "NUM_STAGES": [2, 3],
    "PREFETCH_K": [0, 1, 2],
    "DTYPE": ["fp16","bf16"],
    "CAUSAL": [0],            # non-causal per mission spec
    "QK_CHUNK": [1, 2, 4],
}

ITERS=60
WARMUP=20
SHAPE="mission"
TOPK=3

log_csv = Path("artifacts/tune/tune_log.csv")
topk_json = Path("artifacts/tune/topk.json")
log_csv.parent.mkdir(parents=True, exist_ok=True)

def bench(backend, cfg):
    env = os.environ.copy()
    for k,v in cfg.items():
        env[f"TUNE_{k}"] = str(v)
    # Each trial: run kbench on mission shape
    cmd = [sys.executable, "scripts/kbench.py", "--backend", backend, "--iters", str(ITERS), "--warmup", str(WARMUP), "--shape", SHAPE]
    try:
        out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True)
        # The script prints a JSON summary at the end; parse trailing JSON object
        import re, json
        m = re.findall(r"\{[\s\S]*\}$", out.strip())
        if not m:
            return None, "no-json"
        obj = json.loads(m[-1])
        return obj, None
    except subprocess.CalledProcessError as e:
        return None, f"proc-failed: {e.output[-300:]}"
    except Exception as e:
        return None, str(e)

def main():
    # Log header
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts","backend","config_json","p50_us","correct","notes"])

    results = []
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    space = list(itertools.product(*values))
    print(f"Search space: {len(space)} configs across {len(keys)} knobs")

    for backend in BACKENDS:
        for vals in space:
            cfg = dict(zip(keys, vals))
            ts = time.time()
            obj, err = bench(backend, cfg)
            if obj is None:
                row = [ts, backend, json.dumps(cfg), None, False, f"ERR:{err}"]
                print(f"[FAIL] {backend} {cfg} -> {err}")
            else:
                p50 = obj["p50_us"]
                ok  = bool(obj["numeric_pass"])
                row = [ts, backend, json.dumps(cfg), p50, ok, ""]
                results.append({"backend": backend, "cfg": cfg, "p50_us": p50, "ok": ok})
                print(f"[OK] {backend} {cfg} -> p50={p50:.2f} μs  pass={ok}")
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow(row)

    # Top-K by p50 among passing configs
    winners = [r for r in results if r["ok"]]
    winners.sort(key=lambda x: x["p50_us"])
    topk = winners[:TOPK]
    # Snapshot candidate folders
    out = {"topk": topk, "note": "Top-3 candidates by median latency on mission shape."}
    topk_json.write_text(json.dumps(out, indent=2))

    print(f"Top-{TOPK}:")
    for i,w in enumerate(topk,1):
        print(f"{i}. {w['backend']} {w['cfg']} p50={w['p50_us']:.2f} μs")

if __name__ == "__main__":
    main()
