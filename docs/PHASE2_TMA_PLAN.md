# Phase 2: TMA/Async Memory Integration

**Goal**: 10-20 TFLOPS (15-30× speedup over Phase 1)  
**Method**: Asynchronous memory pipeline with double buffering  
**Hardware**: Hopper TMA (or async copy as fallback)

---

## 🎯 **What is TMA?**

**Tensor Memory Accelerator** (Hopper sm_90+):
- Hardware-accelerated bulk memory transfers
- Asynchronous: GPU can compute while copying
- High bandwidth: Closer to theoretical peak
- Eliminates manual coalescing overhead

**Benefits**:
- **Bandwidth**: 50-70% → 85-90% DRAM utilization
- **Latency**: Overlap memory with compute (free!)
- **Simplicity**: Hardware handles complex addressing

---

## 📈 **Expected Performance Improvement**

### **Phase 1 (Baseline)**
```
DRAM Bandwidth: ~20% (poor - memory-bound)
Compute:        Scalar loops (no Tensor Cores)
Latency:        420ms
TFLOPS:         0.65
```

### **Phase 2 (TMA/Async)**
```
DRAM Bandwidth: 50-70% (2-3× better)
Compute:        Still scalar (Tensor Cores in Phase 3)
Latency:        30-60ms (7-14× faster)
TFLOPS:         10-20 (15-30× speedup)

Why: Memory bandwidth is THE bottleneck!
```

---

## 🏗️ **Implementation Strategy**

### **Approach A: Full TMA (Hopper-specific)**
**Pros**: Maximum bandwidth, hardware-accelerated  
**Cons**: Complex API, CUDA 12.0+ required  
**Code**:
```cuda
// Host: Create TMA descriptor
cudaTensorMapEncodeTiled(&tensorMap, ...);

// Device: Async bulk copy
cp.async.bulk.tensor.2d.shared.global.tile.mbarrier
    [smem_K], [tensorMap, tile_coords], [barrier];
```

### **Approach B: Async Copy (Ampere+)**
**Pros**: Simpler, works on sm_80+  
**Cons**: Less efficient than TMA  
**Code**:
```cuda
// Use cuda::memcpy_async
cuda::memcpy_async(
    smem_K, gmem_K, bytes, pipeline
);
pipeline.producer_commit();
```

### **Phase 2 Choice: Approach B → A**
1. Start with async copy (simpler, proves concept)
2. Measure bandwidth improvement
3. Upgrade to TMA if needed (Phase 2b)

---

## 🔧 **Implementation Checklist**

### **Step 1: Double Buffering**
```cuda
__shared__ __half K_smem[2][BLOCK_N * D];  // Ping-pong
__shared__ __half V_smem[2][BLOCK_N * D];

int read_stage = 0;
int write_stage = 1;

for (tile_n...) {
    // Compute on read_stage while loading write_stage
    compute_QK(K_smem[read_stage]);
    
    // Swap buffers
    read_stage ^= 1;
    write_stage ^= 1;
}
```

### **Step 2: Async Pipeline**
```cuda
#include <cuda/pipeline>

cuda::pipeline<cuda::thread_scope_block> pipe = 
    cuda::make_pipeline();

// Producer: Load K/V
pipe.producer_acquire();
cuda::memcpy_async(smem_K[write_stage], gmem_K, bytes, pipe);
pipe.producer_commit();

// Consumer: Wait for data
pipe.consumer_wait();
compute_QK(smem_K[read_stage]);
pipe.consumer_release();
```

### **Step 3: Shared Memory Increase**
```
Phase 1: 65KB SMEM (1× buffers)
Phase 2: 130KB SMEM (2× buffers for double-buffering)
Budget:  227KB available on H100 ✅
```

### **Step 4: Benchmark**
```bash
# Measure bandwidth
ncu --section MemoryWorkloadAnalysis ./build/bin/test_hopper

# Target metrics
DRAM Throughput: 50-70% (vs 20% in Phase 1)
L2 Hit Rate:     40-60% (reuse across tiles)
Memory Bytes:    Same as Phase 1 (no extra traffic)
```

---

## 📊 **Success Criteria**

| Metric | Phase 1 | Phase 2 Target | Method |
|--------|---------|----------------|--------|
| **Latency** | 420ms | 30-60ms | Async overlap |
| **TFLOPS** | 0.65 | 10-20 | Bandwidth |
| **DRAM %** | ~20% | 50-70% | TMA/async |
| **Correctness** | ✅ | ✅ | No regression |

**Pass**: 10+ TFLOPS AND 50%+ DRAM utilization  
**Excellent**: 15+ TFLOPS AND 60%+ DRAM utilization

---

## 🎓 **References**

### **NVIDIA Docs**
- [Async Copy Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [TMA Documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)
- [cuda::pipeline](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/pipeline.html)

### **FA3 Paper**
- Uses TMA for K/V loads
- Double buffering with 2-4 stages
- Achieves 85-90% DRAM bandwidth

### **CUTLASS 3.x**
- Reference TMA implementations
- [Hopper GEMM Tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/hopper_gemm/README.md)

---

## 🚀 **Next Session**

**Implement**: Phase 2 kernel with async pipeline  
**Test**: On H100, measure DRAM bandwidth with NCU  
**Validate**: 10+ TFLOPS, 50%+ bandwidth, correctness  
**Document**: Performance gains, profiler screenshots

**Ready to code!** 🔥

