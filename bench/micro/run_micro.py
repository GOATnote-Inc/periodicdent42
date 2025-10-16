#!/usr/bin/env python3
import csv, json, os, subprocess, pathlib, sys
from datetime import datetime

root = pathlib.Path(__file__).resolve().parents[2]
binp = root / "bench/micro/bench_many"
logd = root / "evidence"
logd.mkdir(exist_ok=True)
csvp = logd / "micro_log.csv"
bestp = logd / "micro_best.json"

def build(use_tile=False):
    env = os.environ.copy()
    env["CUSTOM_TILE_FLAGS"] = "-DCUSTOM_SDPA_TILE -I src/attn" if use_tile else ""
    subprocess.check_call(["bash", "bench/micro/build_micro.sh"], env=env, cwd=root)

def run(groups=9, tw=4):
    out = subprocess.check_output([str(binp), f"--groups={groups}", f"--tw={tw}", "--csv"], text=True)
    csvp.write_text(out)
    rows = list(csv.DictReader(out.splitlines()))
    for r in rows:
        r["ns_per_iter"] = float(r["ns_per_iter"])
        for k in ("bm", "bk", "stages", "vec"):
            r[k] = int(r[k])
    rows.sort(key=lambda x: x["ns_per_iter"])
    bestp.write_text(json.dumps({"ts": datetime.utcnow().isoformat() + "Z", "topk": rows[:8]}, indent=2))
    print(f"✅ Wrote: {csvp} and {bestp}")
    print(f"\n📊 Top-8 Configurations:")
    print(f"{'Rank':<5} {'BM':<5} {'BK':<5} {'STAGES':<7} {'VEC':<5} {'ns/iter':<10}")
    print("=" * 60)
    for i, r in enumerate(rows[:8], 1):
        print(f"{i:<5} {r['bm']:<5} {r['bk']:<5} {r['stages']:<7} {r['vec']:<5} {r['ns_per_iter']:<10.2f}")

if __name__ == "__main__":
    build(use_tile=os.environ.get("MICRO_USE_TILE", "0") == "1")
    run()

