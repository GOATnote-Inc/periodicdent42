#!/usr/bin/env python3
"""
Parse ptxas output from build logs to extract register/SMEM usage
"""
import re
import sys

def parse_ptxas_log(log_text):
    """
    Extract ptxas metrics from build log
    
    Example output:
    ptxas info    : Used 72 registers, 81920 bytes smem, 400 bytes cmem[0]
    """
    results = []
    
    # Pattern: ptxas info    : Used N registers, M bytes smem
    pattern = r'ptxas info\s+:\s+Used\s+(\d+)\s+registers?,\s+(\d+)\s+bytes\s+smem'
    
    for match in re.finditer(pattern, log_text):
        regs = int(match.group(1))
        smem = int(match.group(2))
        results.append({
            "registers": regs,
            "smem_bytes": smem,
            "smem_kb": smem / 1024.0
        })
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_ptxas.py <build_log.txt>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    with open(log_file, 'r') as f:
        log_text = f.read()
    
    results = parse_ptxas_log(log_text)
    
    if not results:
        print("No ptxas info found in log")
        return
    
    print("PTXAS RESOURCE USAGE")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"Kernel {i+1}:")
        print(f"  Registers:    {r['registers']}")
        print(f"  SMEM:         {r['smem_kb']:.1f} KB ({r['smem_bytes']} bytes)")
        print()
    
    # Check limits (for L4/Ada)
    for i, r in enumerate(results):
        if r['registers'] > 72:
            print(f"⚠️  Kernel {i+1}: High register usage ({r['registers']} > 72)")
        if r['smem_kb'] > 96:
            print(f"⚠️  Kernel {i+1}: High SMEM usage ({r['smem_kb']:.1f} KB > 96 KB)")

if __name__ == "__main__":
    main()


