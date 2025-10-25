#!/usr/bin/env python3
# bench/micro/run_micro.py
import subprocess
import json
import csv
from pathlib import Path

def main():
    repo_root = Path(__file__).parent.parent.parent
    micro_bin = repo_root / "bench" / "micro" / "bench_many"
    
    if not micro_bin.exists():
        print("Building microbench...")
        subprocess.run(["bash", str(repo_root / "bench" / "micro" / "build_micro.sh")], check=True)
    
    print("Running microbench...")
    result = subprocess.run([str(micro_bin)], capture_output=True, text=True)
    
    # Parse output
    lines = result.stdout.strip().split('\n')
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("BLOCK_M,BLOCK_N"):
            header_idx = i
            break
    
    if header_idx is None:
        print("❌ No CSV header found")
        return
    
    # Write CSV
    evidence_dir = repo_root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    
    csv_path = evidence_dir / "micro_log.csv"
    with open(csv_path, 'w') as f:
        f.write('\n'.join(lines[header_idx:]))
    
    print(f"✅ Wrote {csv_path}")
    
    # Parse results (stop at "Top-3" line)
    configs = []
    csv_lines = []
    for line in lines[header_idx:]:
        if line.startswith("Top-3") or line.startswith(" "):
            break
        csv_lines.append(line)
    
    reader = csv.DictReader(csv_lines)
    for row in reader:
        try:
            configs.append({
                "BLOCK_M": int(row["BLOCK_M"]),
                "BLOCK_N": int(row["BLOCK_N"]),
                "VEC_WIDTH": int(row["VEC_WIDTH"]),
                "NUM_WARPS": int(row["NUM_WARPS"]),
                "cycles": float(row["CYCLES"])
            })
        except ValueError:
            continue
    
    # Top-3
    configs.sort(key=lambda x: x["cycles"])
    top3 = configs[:3]
    
    best_path = evidence_dir / "micro_best.json"
    with open(best_path, 'w') as f:
        json.dump(top3, f, indent=2)
    
    print(f"✅ Wrote {best_path}")
    print("\nTop-3:")
    for i, cfg in enumerate(top3):
        print(f"  {i+1}. BLOCK_M={cfg['BLOCK_M']} BLOCK_N={cfg['BLOCK_N']} "
              f"VEC={cfg['VEC_WIDTH']} WARPS={cfg['NUM_WARPS']} "
              f"CYCLES={cfg['cycles']:.0f}")

if __name__ == "__main__":
    main()
