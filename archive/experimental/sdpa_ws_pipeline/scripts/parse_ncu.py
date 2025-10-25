#!/usr/bin/env python3
import argparse, json, subprocess, csv, io, re, pathlib
from pathlib import Path

def import_csv(rep:Path):
    # ncu --import <rep> --csv
    p = subprocess.run(["/usr/local/cuda/bin/ncu","--import",str(rep),"--csv"], capture_output=True, text=True, check=True)
    return p.stdout

def pick(metric_rows, key_regex):
    # return first matching metric's value (float) or None
    for row in metric_rows:
        name = row.get("Metric Name","")
        if re.search(key_regex, name): 
            try: return float(row.get("Metric Value","").replace("%",""))
            except: pass
    return None

def parse_one(rep:Path):
    csvtxt = import_csv(rep)
    # NCU CSV contains multiple tables; pick "Summary" metrics rows
    # We normalize to a list of dicts keyed by 'Metric Name'/'Metric Value'
    metric_rows=[]
    reader = csv.reader(io.StringIO(csvtxt))
    headers=None
    for r in reader:
        if len(r)>=2 and r[0]=="Metric Name":
            headers=r; continue
        if headers and len(r)==len(headers) and r[0] and r[1]:
            metric_rows.append(dict(zip(headers,r)))
    # Derived metrics (best‑effort across arch versions)
    out = {
      "sm_util_pct": pick(metric_rows, r"sm__throughput.*pct_of_peak"),
      "tensor_core_util_pct": pick(metric_rows, r"(hmma|tensor).*sum|tensor.*pct"),
      "achieved_occupancy_pct": pick(metric_rows, r"achieved_occupancy|Occupancy"),
      "eligible_warps_per_cycle": pick(metric_rows, r"warps_eligible.*per_cycle"),
      "l2_hit_rate_pct": pick(metric_rows, r"L2.*hit.*rate|lts__t_sectors.*"),  # heuristic
      "dram_bw_pct_of_peak": pick(metric_rows, r"dram|mem_global.*pct|sectors.*pct"),
      "regs_per_thread": pick(metric_rows, r"registers per thread|Reg/Thr"),
      "kernel_time_us": pick(metric_rows, r"Duration|Time"),
      "warp_stall_breakdown": {}  # filled below from stall metrics if present
    }
    # Aggregate stall metrics (if available)
    for row in metric_rows:
        n = row.get("Metric Name","").lower()
        if "stall" in n and "warp" in n and "pct" in row.get("Metric Value",""):
            out["warp_stall_breakdown"][row["Metric Name"]] = float(row["Metric Value"].replace("%",""))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repdir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repdir = Path(args.repdir); out = {}
    for rep in repdir.glob("*.ncu-rep"):
        out[rep.stem] = parse_one(rep)
    Path(args.out).write_text(json.dumps(out, indent=2))
if __name__=="__main__":
    main()
