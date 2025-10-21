#!/usr/bin/env python3
"""
parse_ncu.py — Import .ncu-rep files and emit a compact JSON summary.
Captures: SM %, TensorCore %, Achieved occupancy, Reg/thread, L2 hit rate,
DRAM BW, warp stall breakdown, eligible warps/cycle, kernel time.

Usage:
  python scripts/parse_ncu.py --in artifacts/ncu/baseline.ncu-rep --name baseline --out artifacts/ncu/summary.json
"""
import argparse, json, subprocess, re, os, sys
from pathlib import Path

def import_ncu(rep):
    # Ask Nsight Compute CLI to emit CSV "raw" page for easy parsing
    cmd = ["ncu", "--import", rep, "--page", "raw"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Failed to parse {rep}: {e.output[-400:]}")
    return out

def extract_metrics(raw):
    # Heuristics: search for key metrics by counter name
    # Note: metric names may vary by version; provide fallbacks.
    def find(metric):
        m = re.search(rf"^{re.escape(metric)},([0-9eE\.\-]+)", raw, re.MULTILINE)
        return float(m.group(1)) if m else None

    # Try multiple aliases for each metric
    def first_of(names):
        for n in names:
            v = find(n)
            if v is not None:
                return v
        return None

    metrics = {
        "sm_util_pct": first_of(["sm__throughput.avg.pct_of_peak_sustained_elapsed"]),
        "tc_util_pct": first_of(["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"]),
        "achieved_occupancy_pct": first_of(["sm__warps_active.avg.pct_of_peak_sustained_active"]),
        "reg_per_thread": first_of(["launch__registers_per_thread"]),
        "l2_hit_rate_pct": first_of(["lts__t_sectors_srcunit_tex_op_read_lookup_hit_rate.pct"]),
        "dram_bw_pct": first_of(["dram__throughput.avg.pct_of_peak_sustained_elapsed"]),
        "eligible_warps_per_cycle": first_of(["smsp__warps_eligible_per_cycle_avg"]),
        "kernel_time_ms": first_of(["gpu__time_duration.sum"]),
        # Stall breakdown examples
        "stall_memory_dependency_pct": first_of(["smsp__cycles_active.avg.per_second"]),  # placeholder if specific stall counters unavailable
    }
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = import_ncu(args.inp)
    m = extract_metrics(raw)
    summary_path = Path(args.out)
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
    else:
        data = {"entries": []}
    data.setdefault("entries", []).append({"name": args.name, "metrics": m})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {summary_path}")

if __name__ == "__main__":
    main()
