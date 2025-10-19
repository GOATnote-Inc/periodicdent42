#!/usr/bin/env python3
"""
NCU Results Parser & I3 Extractor for V2c-v7a Phase 1 Analysis
Goal: Extract EvoEngineer I3 (optimization insights) from NCU profiling data
"""
import csv
import glob
import os
import sys
from pathlib import Path

def parse_ncu_csv(csv_file):
    """Parse NCU CSV output and extract metric values"""
    metrics = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # NCU CSV format: "Metric Name","Metric Unit","Metric Value",...
            metric_name = row.get('Metric Name', row.get('ID', ''))
            metric_value = row.get('Metric Value', row.get('Avg', ''))
            
            if metric_name and metric_value:
                # Try to convert to float, keep as string if fails
                try:
                    metrics[metric_name] = float(metric_value.replace(',', '').replace('%', ''))
                except ValueError:
                    metrics[metric_name] = metric_value
    
    return metrics

def analyze_utilization(metrics):
    """Analyze Profile 1: Tensor Core & Memory Utilization"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 PROFILE 1: Tensor Core & Memory Utilization")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Find metrics (handle different naming conventions)
    tc_active = None
    dram_bw = None
    warp_active = None
    
    for key, val in metrics.items():
        if 'pipe_tensor_active' in key and 'pct' in key:
            tc_active = val
        elif 'dram__throughput' in key and 'pct' in key:
            dram_bw = val
        elif 'warps_active' in key and 'pct' in key:
            warp_active = val
    
    print(f"\n🔹 Tensor Core Utilization: {tc_active:.1f}%" if tc_active else "⚠️  Tensor Core metric not found")
    print(f"🔹 DRAM Throughput: {dram_bw:.1f}%" if dram_bw else "⚠️  DRAM metric not found")
    print(f"🔹 Warp Activity: {warp_active:.1f}%" if warp_active else "⚠️  Warp metric not found")
    
    # I3 Insights
    insights = []
    
    if tc_active is not None:
        if tc_active >= 70:
            insights.append(("✅ TENSOR_CORE_SATURATED", 
                           f"Tensor Cores are highly utilized ({tc_active:.1f}%). "
                           "WMMA compute is NOT the bottleneck. cp.async overlap won't help much."))
        elif tc_active >= 40:
            insights.append(("⚠️  TENSOR_CORE_MODERATE", 
                           f"Tensor Cores moderately utilized ({tc_active:.1f}%). "
                           "Some room for improvement, but not critical."))
        else:
            insights.append(("❌ TENSOR_CORE_UNDERUTILIZED", 
                           f"Tensor Cores underutilized ({tc_active:.1f}%). "
                           "Memory or pipeline stalls are starving WMMA units."))
    
    if dram_bw is not None:
        if dram_bw >= 80:
            insights.append(("❌ MEMORY_BANDWIDTH_BOUND", 
                           f"DRAM bandwidth saturated ({dram_bw:.1f}%). "
                           "Memory is the bottleneck. cp.async overlap SHOULD help but isn't."))
        elif dram_bw >= 50:
            insights.append(("⚠️  MEMORY_BANDWIDTH_MODERATE", 
                           f"DRAM bandwidth moderate ({dram_bw:.1f}%). "
                           "Memory not a critical bottleneck."))
        else:
            insights.append(("✅ MEMORY_BANDWIDTH_OK", 
                           f"DRAM bandwidth low ({dram_bw:.1f}%). "
                           "Memory is NOT the bottleneck. cp.async overlap won't help."))
    
    return insights

def analyze_async(metrics):
    """Analyze Profile 2: cp.async & Memory Pipeline"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 PROFILE 2: cp.async & Memory Pipeline")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    cp_async_ops = None
    mem_128b = None
    mem_64b = None
    
    for key, val in metrics.items():
        if 'cp_async' in key and 'sum' in key:
            cp_async_ops = val
        elif 'memory_128b' in key and 'sum' in key:
            mem_128b = val
        elif 'memory_64b' in key and 'sum' in key:
            mem_64b = val
    
    print(f"\n🔹 cp.async Operations: {cp_async_ops:,.0f}" if cp_async_ops else "⚠️  cp.async metric not found")
    print(f"🔹 128B Memory Ops: {mem_128b:,.0f}" if mem_128b else "⚠️  128B ops metric not found")
    print(f"🔹 64B Memory Ops: {mem_64b:,.0f}" if mem_64b else "⚠️  64B ops metric not found")
    
    insights = []
    
    if cp_async_ops is not None:
        if cp_async_ops == 0:
            insights.append(("❌ CP_ASYNC_NOT_USED", 
                           "cp.async is NOT being used! All loads falling back to __ldg(). "
                           "Check alignment or cp_async_16B_if_aligned() logic."))
        elif cp_async_ops > 0:
            insights.append(("✅ CP_ASYNC_ACTIVE", 
                           f"cp.async is active ({cp_async_ops:,.0f} ops). "
                           "But overlap may not be effective if wait_group is too conservative."))
    
    if mem_128b and mem_64b:
        ratio = mem_128b / (mem_128b + mem_64b + 1e-10)
        if ratio >= 0.8:
            insights.append(("✅ COALESCED_LOADS", 
                           f"Most loads are 128B ({ratio*100:.1f}%). Memory access is well-coalesced."))
        else:
            insights.append(("⚠️  UNCOALESCED_LOADS", 
                           f"Many non-128B loads ({(1-ratio)*100:.1f}%). Consider vectorization."))
    
    return insights

def analyze_smem(metrics):
    """Analyze Profile 3: SMEM Bank Conflicts & Occupancy"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 PROFILE 3: SMEM Bank Conflicts & Occupancy")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    bank_conflicts = None
    occupancy_pct = None
    regs_per_thread = None
    smem_bytes = None
    
    for key, val in metrics.items():
        if 'bank_conflicts' in key and 'sum' in key and 'ld' not in key and 'st' not in key:
            bank_conflicts = val
        elif 'maximum_warps_per_active_cycle' in key:
            occupancy_pct = val
        elif 'registers_per_thread' in key:
            regs_per_thread = val
        elif 'shared_mem_per_block' in key and 'dynamic' in key:
            smem_bytes = val
    
    print(f"\n🔹 SMEM Bank Conflicts: {bank_conflicts:,.0f}" if bank_conflicts is not None else "⚠️  Bank conflicts metric not found")
    print(f"🔹 Occupancy: {occupancy_pct:.1f}%" if occupancy_pct else "⚠️  Occupancy metric not found")
    print(f"🔹 Registers/Thread: {regs_per_thread:.0f}" if regs_per_thread else "⚠️  Regs metric not found")
    print(f"🔹 Dynamic SMEM: {smem_bytes/1024:.1f} KB" if smem_bytes else "⚠️  SMEM metric not found")
    
    insights = []
    
    if bank_conflicts is not None:
        if bank_conflicts > 10000:
            insights.append(("❌ HIGH_BANK_CONFLICTS", 
                           f"Significant SMEM bank conflicts ({bank_conflicts:,.0f}). "
                           "Apply XOR swizzle to K^T layout (Phase 2)."))
        elif bank_conflicts > 1000:
            insights.append(("⚠️  MODERATE_BANK_CONFLICTS", 
                           f"Some SMEM bank conflicts ({bank_conflicts:,.0f}). "
                           "XOR swizzle may provide modest gain."))
        else:
            insights.append(("✅ LOW_BANK_CONFLICTS", 
                           f"Minimal SMEM bank conflicts ({bank_conflicts:,.0f}). "
                           "XOR swizzle won't help significantly."))
    
    if occupancy_pct is not None:
        if occupancy_pct >= 75:
            insights.append(("✅ HIGH_OCCUPANCY", 
                           f"Good occupancy ({occupancy_pct:.1f}%). Sufficient warps to hide latency."))
        elif occupancy_pct >= 50:
            insights.append(("⚠️  MODERATE_OCCUPANCY", 
                           f"Moderate occupancy ({occupancy_pct:.1f}%). May benefit from reducing regs/SMEM."))
        else:
            insights.append(("❌ LOW_OCCUPANCY", 
                           f"Low occupancy ({occupancy_pct:.1f}%). Reduce registers or SMEM usage."))
    
    return insights

def analyze_stalls(metrics):
    """Analyze Profile 4: Stall Analysis"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 PROFILE 4: Stall Analysis")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    stalls = {}
    for key, val in metrics.items():
        if 'warp_issue_stalled' in key and 'pct' in key:
            stall_type = key.split('_stalled_')[1].split('_per')[0]
            stalls[stall_type] = val
    
    if stalls:
        print("\n🔹 Stall Breakdown:")
        for stall_type, pct in sorted(stalls.items(), key=lambda x: x[1], reverse=True):
            print(f"   {stall_type:30s}: {pct:6.2f}%")
    else:
        print("\n⚠️  Stall metrics not found")
    
    insights = []
    
    # Find dominant stall
    if stalls:
        dominant_stall = max(stalls.items(), key=lambda x: x[1])
        stall_name, stall_pct = dominant_stall
        
        if stall_pct >= 20:
            if 'long_scoreboard' in stall_name:
                insights.append(("❌ MEMORY_LATENCY_STALLS", 
                               f"Dominant stall: long_scoreboard ({stall_pct:.1f}%). "
                               "Memory latency is the issue. cp.async overlap SHOULD help but wait_group may be blocking."))
            elif 'barrier' in stall_name:
                insights.append(("❌ BARRIER_STALLS", 
                               f"Dominant stall: barrier ({stall_pct:.1f}%). "
                               "Too many __syncthreads(). Reduce synchronization points."))
            elif 'math_pipe_throttle' in stall_name:
                insights.append(("⚠️  MATH_PIPE_THROTTLE", 
                               f"Dominant stall: math_pipe_throttle ({stall_pct:.1f}%). "
                               "Compute units throttled. May need better ILP or instruction mix."))
            else:
                insights.append(("⚠️  OTHER_STALLS", 
                               f"Dominant stall: {stall_name} ({stall_pct:.1f}%)."))
    
    return insights

def generate_i3_report(all_insights):
    """Generate EvoEngineer I3 (optimization insights) report"""
    print("\n")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎯 I3: OPTIMIZATION INSIGHTS (EvoEngineer)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    for tag, insight in all_insights:
        status = "✅" if tag.startswith("✅") else ("❌" if tag.startswith("❌") else "⚠️ ")
        print(f"\n{status} {tag}")
        print(f"   {insight}")
    
    print("\n")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📋 RECOMMENDATION")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Analyze insights and make recommendation
    has_tc_saturated = any('TENSOR_CORE_SATURATED' in tag for tag, _ in all_insights)
    has_mem_ok = any('MEMORY_BANDWIDTH_OK' in tag for tag, _ in all_insights)
    has_high_conflicts = any('HIGH_BANK_CONFLICTS' in tag for tag, _ in all_insights)
    has_cp_async_issue = any('CP_ASYNC_NOT_USED' in tag for tag, _ in all_insights)
    
    print("\nBased on NCU profiling:")
    print()
    
    if has_tc_saturated and has_mem_ok:
        print("✅ **ACCEPT v6a/v7a as BEST CUSTOM KERNEL**")
        print()
        print("   Tensor Cores are saturated and memory is not the bottleneck.")
        print("   cp.async overlap cannot help because compute is already fast.")
        print("   Phases 2-4 (swizzle, fusion, persistent CTAs) unlikely to improve.")
        print()
        print("   V2c-v6a GREEN: 1177 μs (100% correct, 1.68× from scalar)")
        print("   Status: Research artifact demonstrating EvoEngineer methodology")
        print()
        print("   For production: Use PyTorch SDPA (31 μs, 38× faster)")
    
    elif has_cp_async_issue:
        print("⚠️  **FIX cp.async ALIGNMENT ISSUE**")
        print()
        print("   cp.async is not being used (falling back to __ldg).")
        print("   Check 16B alignment in cp_async_16B_if_aligned().")
        print("   Once fixed, re-measure to see if overlap helps.")
    
    elif has_high_conflicts:
        print("⚠️  **TRY PHASE 2: XOR Swizzle**")
        print()
        print("   High SMEM bank conflicts detected.")
        print("   Apply XOR swizzle to K^T layout may provide 1.2-1.5× gain.")
        print("   Worth trying before accepting result.")
    
    else:
        print("⚠️  **MIXED RESULTS - TARGETED FIXES**")
        print()
        print("   No clear single bottleneck identified.")
        print("   Consider:")
        print("   - Adjust wait_group<> count (may be too conservative)")
        print("   - Increase producer warps (single warp may not saturate)")
        print("   - Profile with longer sequences (L=4096) to see if overlap helps")
    
    print()

def main():
    ncu_dir = Path("ncu_results")
    
    if not ncu_dir.exists():
        print("❌ ncu_results/ directory not found. Run profile_ncu.sh first.")
        return 1
    
    # Find latest profile files
    profile_files = {
        1: sorted(glob.glob(str(ncu_dir / "profile1_utilization_*.csv")))[-1:],
        2: sorted(glob.glob(str(ncu_dir / "profile2_async_*.csv")))[-1:],
        3: sorted(glob.glob(str(ncu_dir / "profile3_smem_occupancy_*.csv")))[-1:],
        4: sorted(glob.glob(str(ncu_dir / "profile4_stalls_*.csv")))[-1:],
    }
    
    all_insights = []
    
    for profile_id, files in profile_files.items():
        if not files:
            print(f"⚠️  Profile {profile_id} not found, skipping...")
            continue
        
        metrics = parse_ncu_csv(files[0])
        
        if profile_id == 1:
            all_insights.extend(analyze_utilization(metrics))
        elif profile_id == 2:
            all_insights.extend(analyze_async(metrics))
        elif profile_id == 3:
            all_insights.extend(analyze_smem(metrics))
        elif profile_id == 4:
            all_insights.extend(analyze_stalls(metrics))
    
    # Generate I3 report
    generate_i3_report(all_insights)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

