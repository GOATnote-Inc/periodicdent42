#!/usr/bin/env python3
"""
Roofline Plot Generator for Shadow Nsight Reports
Reads JSON benchmark data and plots compute vs memory bound position
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_report(json_path):
    """Load Shadow Nsight JSON report"""
    with open(json_path) as f:
        return json.load(f)

def plot_roofline(report, output_path='reports/roofline.png'):
    """Generate roofline plot from benchmark data"""
    
    # H100 specifications
    peak_fp16_tflops = 1979.0  # FP16 with Tensor Cores
    peak_dram_gbs = 3350.0     # HBM3 bandwidth
    
    # Extract measured values
    tflops = report['performance']['tflops']
    gbs = report['performance']['gbs']
    
    # Compute arithmetic intensity (ops/byte)
    ai = (tflops * 1e12) / (gbs * 1e9)
    
    # Roofline calculation
    ai_range = np.logspace(-2, 3, 1000)
    roofline = np.minimum(
        peak_fp16_tflops * np.ones_like(ai_range),
        ai_range * peak_dram_gbs / 1000  # GB/s to TB/s
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot roofline
    ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='H100 Roofline')
    
    # Mark peak compute and memory limits
    ax.axhline(y=peak_fp16_tflops, color='gray', linestyle='--', alpha=0.5, label='Peak FP16 Compute')
    ax.axvline(x=peak_fp16_tflops / (peak_dram_gbs / 1000), color='gray', linestyle='--', alpha=0.5)
    
    # Plot measured kernel
    ax.plot(ai, tflops, 'ro', markersize=15, label=f'{report["kernel"]} (measured)', zorder=10)
    
    # Add annotations
    ax.annotate(
        f'{tflops:.1f} TFLOPS\n{gbs:.1f} GB/s\nAI = {ai:.2f}',
        xy=(ai, tflops),
        xytext=(ai * 1.5, tflops * 0.7),
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3')
    )
    
    # Determine bottleneck
    ridge_point = peak_fp16_tflops / (peak_dram_gbs / 1000)
    if ai < ridge_point:
        bottleneck = "Memory Bound"
        color = 'red'
    else:
        bottleneck = "Compute Bound"
        color = 'green'
    
    ax.text(0.02, 0.98, f'Status: {bottleneck}', 
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
    ax.set_title(f'Roofline Analysis: {report["kernel"]}\n{report["device"]}', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # Set limits
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(1, peak_fp16_tflops * 1.2)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Roofline plot saved to: {output_path}")
    
    # Print analysis
    print(f"\n📊 Roofline Analysis:")
    print(f"   Arithmetic Intensity: {ai:.2f} FLOPs/Byte")
    print(f"   Ridge Point: {ridge_point:.2f} FLOPs/Byte")
    print(f"   Status: {bottleneck}")
    
    if ai < ridge_point:
        improvement = ridge_point / ai
        print(f"   💡 Increase AI by {improvement:.2f}× to become compute-bound")
    else:
        improvement = peak_fp16_tflops / tflops
        print(f"   💡 Can improve by {improvement:.2f}× (compute headroom)")
    
    return ai, bottleneck

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_roofline.py <json_report>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'reports/roofline.png'
    
    # Create reports directory if needed
    Path('reports').mkdir(exist_ok=True)
    
    # Load and plot
    report = load_report(json_path)
    plot_roofline(report, output_path)

if __name__ == '__main__':
    main()

