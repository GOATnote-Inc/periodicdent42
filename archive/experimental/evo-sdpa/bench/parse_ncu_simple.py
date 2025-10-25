#!/usr/bin/env python3
"""
Simplified NCU Results Parser for V2c-v7a Essential Metrics
Focuses on key metrics to quickly determine bottleneck
"""
import glob
import re
import sys
from pathlib import Path

def extract_metrics_from_csv(csv_file):
    """Extract metrics from NCU CSV output"""
    metrics = {}
    
    with open(csv_file, 'r') as f:
        content = f.read()
        
        # NCU CSV format varies, try to extract key metrics
        # Pattern: "Metric Name","Unit","Value"
        
        # Tensor Core utilization
        match = re.search(r'sm__pipe_tensor_active[^,]*,[^,]*,([0-9.]+)', content)
        if match:
            metrics['tc_active'] = float(match.group(1))
        
        # DRAM throughput
        match = re.search(r'dram__throughput[^,]*pct[^,]*,[^,]*,([0-9.]+)', content)
        if match:
            metrics['dram_bw'] = float(match.group(1))
        
        # cp.async operations
        match = re.search(r'cp_async\.sum[^,]*,[^,]*,([0-9,]+)', content)
        if match:
            metrics['cp_async_ops'] = float(match.group(1).replace(',', ''))
        
        # SMEM bank conflicts
        match = re.search(r'bank_conflicts[^,]*\.sum[^,]*,[^,]*,([0-9,]+)', content)
        if match:
            metrics['bank_conflicts'] = float(match.group(1).replace(',', ''))
        
        # Warp activity
        match = re.search(r'warps_active[^,]*pct[^,]*,[^,]*,([0-9.]+)', content)
        if match:
            metrics['warp_active'] = float(match.group(1))
        
        # Registers per thread
        match = re.search(r'registers_per_thread[^,]*,[^,]*,([0-9.]+)', content)
        if match:
            metrics['regs_per_thread'] = float(match.group(1))
        
        # Dynamic SMEM
        match = re.search(r'shared_mem_per_block_dynamic[^,]*,[^,]*,([0-9,]+)', content)
        if match:
            metrics['smem_bytes'] = float(match.group(1).replace(',', ''))
        
        # Long scoreboard stalls (memory latency)
        match = re.search(r'long_scoreboard[^,]*pct[^,]*,[^,]*,([0-9.]+)', content)
        if match:
            metrics['mem_stall_pct'] = float(match.group(1))
        
        # Barrier stalls
        match = re.search(r'barrier[^,]*pct[^,]*,[^,]*,([0-9.]+)', content)
        if match:
            metrics['barrier_stall_pct'] = float(match.group(1))
    
    return metrics

def analyze_and_recommend(metrics):
    """Analyze metrics and provide recommendation"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 V2c-v7a NCU PROFILING RESULTS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Display metrics
    print("\n🔹 Key Metrics:")
    print()
    if 'tc_active' in metrics:
        print(f"  Tensor Core Utilization:  {metrics['tc_active']:6.2f}%")
    if 'dram_bw' in metrics:
        print(f"  DRAM Throughput:          {metrics['dram_bw']:6.2f}%")
    if 'warp_active' in metrics:
        print(f"  Warp Activity:            {metrics['warp_active']:6.2f}%")
    if 'cp_async_ops' in metrics:
        print(f"  cp.async Operations:      {metrics['cp_async_ops']:,.0f}")
    if 'bank_conflicts' in metrics:
        print(f"  SMEM Bank Conflicts:      {metrics['bank_conflicts']:,.0f}")
    if 'regs_per_thread' in metrics:
        print(f"  Registers/Thread:         {metrics['regs_per_thread']:.0f}")
    if 'smem_bytes' in metrics:
        print(f"  Dynamic SMEM:             {metrics['smem_bytes']/1024:.1f} KB")
    if 'mem_stall_pct' in metrics:
        print(f"  Memory Latency Stalls:    {metrics['mem_stall_pct']:6.2f}%")
    if 'barrier_stall_pct' in metrics:
        print(f"  Barrier Stalls:           {metrics['barrier_stall_pct']:6.2f}%")
    
    # Analysis
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎯 ANALYSIS & RECOMMENDATION")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    tc_active = metrics.get('tc_active', 0)
    dram_bw = metrics.get('dram_bw', 0)
    cp_async_ops = metrics.get('cp_async_ops', 0)
    bank_conflicts = metrics.get('bank_conflicts', 0)
    mem_stall = metrics.get('mem_stall_pct', 0)
    barrier_stall = metrics.get('barrier_stall_pct', 0)
    
    # Determine bottleneck
    print("🔍 Bottleneck Analysis:")
    print()
    
    if tc_active >= 70:
        print("✅ TENSOR CORES SATURATED ({:.1f}%)".format(tc_active))
        print("   → WMMA compute is NOT the bottleneck")
        print("   → cp.async overlap won't help (compute already fast)")
        recommendation = "ACCEPT_V6A"
    elif dram_bw >= 75:
        print("❌ MEMORY BANDWIDTH BOUND ({:.1f}%)".format(dram_bw))
        print("   → Memory is the bottleneck")
        if cp_async_ops == 0:
            print("   → cp.async NOT being used! (alignment issue)")
            recommendation = "FIX_CP_ASYNC"
        else:
            print("   → cp.async is active but wait_group may be too conservative")
            recommendation = "TUNE_WAIT_GROUP"
    elif bank_conflicts > 10000:
        print("❌ HIGH SMEM BANK CONFLICTS ({:,.0f})".format(bank_conflicts))
        print("   → SMEM layout is causing contention")
        recommendation = "TRY_SWIZZLE"
    elif barrier_stall >= 15:
        print("❌ HIGH BARRIER STALLS ({:.1f}%)".format(barrier_stall))
        print("   → Too many __syncthreads()")
        recommendation = "REDUCE_SYNC"
    elif mem_stall >= 20:
        print("⚠️  MEMORY LATENCY STALLS ({:.1f}%)".format(mem_stall))
        print("   → Memory latency is an issue")
        if cp_async_ops == 0:
            recommendation = "FIX_CP_ASYNC"
        else:
            recommendation = "TUNE_WAIT_GROUP"
    elif tc_active < 40:
        print("⚠️  TENSOR CORES UNDERUTILIZED ({:.1f}%)".format(tc_active))
        print("   → WMMA units are starved")
        recommendation = "INVESTIGATE_STARVATION"
    else:
        print("⚠️  MIXED BOTTLENECK")
        print("   → No dominant bottleneck")
        recommendation = "ACCEPT_V6A"
    
    # Final recommendation
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📋 RECOMMENDATION")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    if recommendation == "ACCEPT_V6A":
        print("✅ **ACCEPT V2c-v6a/v7a as BEST CUSTOM KERNEL**")
        print()
        print("   Tensor Cores are saturated. WMMA compute is already efficient.")
        print("   cp.async overlap cannot help because compute is the limiting factor.")
        print()
        print("   V2c-v6a GREEN: 1177 μs (100% correct, 1.68× from scalar)")
        print("   V2c-v7a: 1162 μs (100% correct, 1.01× from v6a)")
        print()
        print("   Phases 2-4 (swizzle, fusion, persistent CTAs) will NOT close the")
        print("   38× gap to PyTorch SDPA (31 μs).")
        print()
        print("   ✅ Use PyTorch SDPA or xFormers CUTLASS for production")
        print("   ✅ Document custom kernel as research artifact")
    
    elif recommendation == "FIX_CP_ASYNC":
        print("❌ **FIX cp.async ALIGNMENT ISSUE**")
        print()
        print("   cp.async is NOT being used ({:,.0f} ops).".format(cp_async_ops))
        print("   All loads are falling back to __ldg().")
        print()
        print("   Action: Debug cp_async_16B_if_aligned() logic")
        print("   Check: 16B alignment for both SMEM and global pointers")
        print("   Expected: cp.async ops should be > 0 if working")
    
    elif recommendation == "TRY_SWIZZLE":
        print("⚠️  **TRY PHASE 2: XOR Swizzle**")
        print()
        print("   High SMEM bank conflicts detected ({:,.0f}).".format(bank_conflicts))
        print("   Apply XOR swizzle to K^T layout may provide 1.2-1.5× gain.")
        print()
        print("   Estimated effort: 2-3 hours")
        print("   Expected speedup: 1.2-1.5× (1162 μs → 800-950 μs)")
        print("   Worth trying before accepting result")
    
    elif recommendation == "REDUCE_SYNC":
        print("⚠️  **REDUCE SYNCHRONIZATION OVERHEAD**")
        print()
        print("   Barrier stalls are high ({:.1f}%).".format(barrier_stall))
        print("   Too many __syncthreads() calls blocking progress.")
        print()
        print("   Action: Review kernel for unnecessary barriers")
        print("   Consider: Reduce stage ring hand-off sync points")
    
    elif recommendation == "TUNE_WAIT_GROUP":
        print("⚠️  **TUNE cp.async wait_group<> COUNT**")
        print()
        print("   Memory latency is an issue, but cp.async is active.")
        print("   wait_group<> may be too conservative (waiting too early).")
        print()
        print("   Current: cp_async_wait_group<STAGES-1>()")
        print("   Try: Experiment with different wait counts")
        print("   Goal: Allow more overlap between async and compute")
    
    else:
        print("⚠️  **CONTINUE INVESTIGATION**")
        print()
        print("   Mixed or unclear bottleneck.")
        print("   Consider more detailed profiling or accept current result.")
    
    print()

def main():
    ncu_dir = Path("ncu_results")
    
    if not ncu_dir.exists():
        print("❌ ncu_results/ directory not found. Run profile_ncu_simple.sh first.")
        return 1
    
    # Find latest profile
    profile_files = sorted(glob.glob(str(ncu_dir / "ncu_essential_*.csv")))
    
    if not profile_files:
        print("❌ No NCU results found. Run profile_ncu_simple.sh first.")
        return 1
    
    latest_file = profile_files[-1]
    print(f"\n📁 Parsing: {latest_file}")
    
    metrics = extract_metrics_from_csv(latest_file)
    
    if not metrics:
        print("⚠️  No metrics extracted. Check NCU output format.")
        print("\nRaw file content (first 50 lines):")
        with open(latest_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                print(line.rstrip())
        return 1
    
    analyze_and_recommend(metrics)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

