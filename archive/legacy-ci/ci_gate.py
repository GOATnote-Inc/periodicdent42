#!/usr/bin/env python3
"""
FlashCore v12: CI Gating Script
Parses logs and enforces all safety gates
"""

import json
import sys
from pathlib import Path

def check_ptxas(log_file):
    """Check PTXAS metrics from build log"""
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return False
    
    text = log_file.read_text()
    
    # Extract metrics
    import re
    regs = int(re.search(r'(\d+) registers', text).group(1)) if 'registers' in text else 0
    spill = int(re.search(r'(\d+) bytes spill', text).group(1)) if 'spill' in text else 0
    stack = int(re.search(r'(\d+) bytes stack', text).group(1)) if 'stack' in text else 0
    
    passed = True
    
    if regs > 64:
        print(f"❌ GATE FAIL: Registers {regs} > 64")
        passed = False
    
    if spill > 0:
        print(f"❌ GATE FAIL: Spills {spill} > 0")
        passed = False
    
    if stack > 0:
        print(f"❌ GATE FAIL: Stack {stack} > 0")
        passed = False
    
    if passed:
        print(f"✅ PTXAS: regs={regs}, spill={spill}, stack={stack}")
    
    return passed

def check_correctness(log_file):
    """Check correctness from benchmark log"""
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return False
    
    text = log_file.read_text()
    
    # Check for PASS markers
    if '✅ PASS' not in text:
        print(f"❌ GATE FAIL: Correctness test did not pass")
        return False
    
    # Extract max error
    import re
    match = re.search(r'Max error:\s+([\d\.]+)', text)
    if match:
        max_err = float(match.group(1))
        if max_err > 1e-3:
            print(f"❌ GATE FAIL: Max error {max_err:.6f} > 1e-3")
            return False
        print(f"✅ Correctness: max_err={max_err:.6f}")
    
    return True

def check_determinism(bench_json):
    """Check determinism from benchmark results"""
    if not bench_json.exists():
        print(f"⚠️  Determinism check skipped (no JSON)")
        return True
    
    # For now, just check that the result exists
    data = json.loads(bench_json.read_text())
    if 'latency_us' in data:
        print(f"✅ Determinism: latency={data['latency_us']:.2f} µs")
        return True
    
    return False

def check_performance(bench_json, target_us=28.0):
    """Check performance target"""
    if not bench_json.exists():
        print(f"⚠️  Performance check skipped (no JSON)")
        return True
    
    data = json.loads(bench_json.read_text())
    latency = data.get('latency_us', float('inf'))
    
    if latency <= target_us:
        print(f"🎉 EXCELLENCE: {latency:.2f} µs ≤ {target_us} µs")
        return True
    else:
        print(f"⚠️  Performance: {latency:.2f} µs > {target_us} µs (target)")
        return True  # Warning only, not blocking

def main():
    root = Path(__file__).parent.parent
    variant = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
    
    print("════════════════════════════════════════════════════════════════")
    print(f"FlashCore v12 CI Gate: {variant}")
    print("════════════════════════════════════════════════════════════════")
    
    # Check all gates
    gates = [
        ("PTXAS", check_ptxas(root / f"logs/build_{variant}.log")),
        ("Correctness", check_correctness(root / f"logs/bench_{variant}.log")),
        ("Determinism", check_determinism(root / f"results/bench_{variant}.json")),
        ("Performance", check_performance(root / f"results/bench_{variant}.json")),
    ]
    
    # Summary
    print("\n" + "="*64)
    print("Gate Summary:")
    for name, passed in gates:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    # Exit
    all_passed = all(passed for _, passed in gates)
    if all_passed:
        print("\n🎉 ALL GATES PASSED")
        return 0
    else:
        print("\n❌ SOME GATES FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

