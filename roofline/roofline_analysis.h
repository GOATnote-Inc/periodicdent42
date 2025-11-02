// Roofline Analysis Framework
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

struct RooflineMetrics {
    double flops;           // Total FLOPs
    double bytes_dram;      // Bytes moved from DRAM
    double time_ms;         // Kernel time
    double oi;              // Operational intensity (FLOPs/byte)
    double tflops;          // Achieved TFLOP/s
    double bw_gbs;          // Achieved BW (GB/s)
    bool memory_bound;      // True if memory-bound
    
    void compute(double flops_, double bytes_, double ms_) {
        flops = flops_;
        bytes_dram = bytes_;
        time_ms = ms_;
        
        oi = flops / bytes_dram;
        tflops = (flops / (ms_ / 1000.0)) / 1e12;
        bw_gbs = (bytes_dram / (ms_ / 1000.0)) / 1e9;
        
        // H100: 3958 TFLOP/s FP16, 3350 GB/s HBM3
        double peak_tflops = 3958.0;
        double peak_bw = 3350.0;
        double balance_point = peak_tflops / peak_bw;  // ~1.18 FLOP/byte
        
        memory_bound = (oi < balance_point);
    }
    
    void print(const char* name) {
        printf("═══════════════════════════════════════════════════════════\n");
        printf("  %s Roofline Analysis\n", name);
        printf("═══════════════════════════════════════════════════════════\n");
        printf("FLOPs:          %.3e\n", flops);
        printf("Bytes (DRAM):   %.3e (%.2f GB)\n", bytes_dram, bytes_dram/1e9);
        printf("Time:           %.3f ms\n", time_ms);
        printf("\n");
        printf("Operational Intensity:  %.1f FLOPs/byte\n", oi);
        printf("Achieved TFLOP/s:       %.1f\n", tflops);
        printf("Achieved BW:            %.1f GB/s\n", bw_gbs);
        printf("\n");
        printf("H100 Limits:\n");
        printf("  Peak TFLOP/s (FP16):  3958\n");
        printf("  Peak BW (HBM3):       3350 GB/s\n");
        printf("  Balance point:        1.18 FLOP/byte\n");
        printf("\n");
        if (memory_bound) {
            printf("⚠️  MEMORY-BOUND (OI < 1.18)\n");
            printf("    → Optimize: TMA, tiling, fusion, on-chip residency\n");
            printf("    → Current OI: %.1f FLOP/byte\n", oi);
            printf("    → Wasted compute: %.1f%%\n", 
                   100.0 * (1.0 - bw_gbs / 3350.0));
        } else {
            printf("⚠️  COMPUTE-BOUND (OI >= 1.18)\n");
            printf("    → Optimize: tensor cores, pipeline, register pressure\n");
            printf("    → Current util: %.1f%% of peak\n", 100.0 * tflops / 3958.0);
        }
        printf("═══════════════════════════════════════════════════════════\n");
    }
};

// GEMM FLOPs
inline double gemm_flops(int M, int N, int K) {
    return 2.0 * (double)M * (double)N * (double)K;
}

// GEMM bytes (conservative: A + B + C_write, no beta)
inline double gemm_bytes_fp16(int M, int N, int K) {
    double bytes_A = (double)M * K * 2;  // FP16
    double bytes_B = (double)K * N * 2;
    double bytes_C = (double)M * N * 2;
    return bytes_A + bytes_B + bytes_C;
}

// Attention FLOPs (QK^T + softmax + PV)
inline double attention_flops(int B, int H, int S, int D) {
    double qk = 2.0 * B * H * S * S * D;       // Q @ K^T
    double softmax = B * H * S * S * 3.0;      // exp + sum + div
    double pv = 2.0 * B * H * S * S * D;       // P @ V
    return qk + softmax + pv;
}

// Attention bytes (Q + K + V + O, no intermediate)
inline double attention_bytes_fp16(int B, int H, int S, int D) {
    double bytes_qkv = 3.0 * B * S * H * D * 2;  // Q, K, V
    double bytes_o = B * S * H * D * 2;           // Output
    return bytes_qkv + bytes_o;
}
