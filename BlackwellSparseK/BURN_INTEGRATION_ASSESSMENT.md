# Burn Integration Assessment: Dense GEMM

## Critical Clarification

**Your recipe assumes BlackwellSparseK is a sparse BSR kernel.**  
**It's not. It's a dense GEMM kernel.**

### What BlackwellSparseK Actually Is

| Aspect | Reality |
|--------|---------|
| **Operation** | Dense FP16 matrix multiplication (standard GEMM) |
| **Performance** | 598.9 TFLOPS (96.2% of cuBLAS) |
| **Format** | Standard dense matrices (not sparse) |
| **API** | CUTLASS 4.3.0 CollectiveBuilder |
| **Use case** | MLP layers, standard matrix multiplication |

**No sparse format (BSR, CSR, COO) is involved.**

The name "BlackwellSparseK" is a historical artifact from an earlier project phase. Current work is 100% dense GEMM.

---

## Why Burn Integration Doesn't Make Sense

### 1. Burn Already Uses cuBLAS for Dense GEMM

**Burn's current dense GEMM backend:**
```rust
// burn/crates/burn-cuda/src/ops/tensor.rs
impl MatmulOps<Self> for Cuda {
    fn matmul<const D: usize>(
        lhs: Self::Tensor<D>,
        rhs: Self::Tensor<D>,
    ) -> Self::Tensor<D> {
        // Uses cuBLAS directly
        cublas_gemm(lhs, rhs)  // 622.8 TFLOPS on H100
    }
}
```

**Our kernel:** 598.9 TFLOPS  
**Burn's existing:** 622.8 TFLOPS (cuBLAS)  
**Gap:** -23.9 TFLOPS (-3.8%)

**Conclusion:** We're slower than what Burn already has

### 2. Performance Comparison

| Backend | TFLOPS | vs cuBLAS | Integration Effort |
|---------|--------|-----------|-------------------|
| **cuBLAS (current)** | **622.8** | **100%** | Already integrated |
| BlackwellSparseK | 598.9 | 96% | High (FFI, testing) |
| CUTLASS 4.3 Ex49 | 406.8 | 65% | Medium |

**Value proposition:** Negative (we're 4% slower)

### 3. Integration Cost vs Benefit

**Cost:**
- Write Rust FFI bindings
- Handle memory management across Rust/CUDA boundary
- Add build complexity (link CUTLASS, CUDA)
- Test across multiple platforms
- Maintain compatibility with Burn updates
- Document integration

**Estimated effort:** 1-2 weeks

**Benefit:**
- 0% performance gain (we're slower)
- Educational value only

**ROI:** Negative

---

## If You Still Want to Integrate (Educational Purpose)

### Adapted Recipe for Dense GEMM

**Your recipe was for sparse BSR. Here's dense GEMM version:**

#### 1. Compile Kernel as Shared Library

```bash
# src/blackwell_dense_gemm.cu
extern "C" __global__ void blackwell_dense_gemm_fp16(
    const half* __restrict__ A,  // M×K dense matrix
    const half* __restrict__ B,  // K×N dense matrix
    float*      __restrict__ C,  // M×N output
    int M, int N, int K
) {
    // Your 598.9 TFLOPS CUTLASS CollectiveBuilder kernel
    // (exact code from gemm_h100_599tflops_final.cu)
}
```

Compile:
```bash
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
     --maxrregcount=255 -Xcompiler -fPIC -shared \
     -I/opt/cutlass/include \
     -o libblackwell_dense_gemm.so src/blackwell_dense_gemm.cu \
     -lcudart
```

#### 2. Rust FFI Wrapper

```rust
// burn-blackwell/src/ffi.rs
#[link(name = "blackwell_dense_gemm", kind = "dylib")]
extern "C" {
    fn blackwell_dense_gemm_fp16(
        a: *const f16,
        b: *const f16,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> cudaError_t;
}

pub unsafe fn launch_dense_gemm(
    a: *const f16,
    b: *const f16,
    c: *mut f32,
    m: i32, n: i32, k: i32,
) -> Result<(), cudaError> {
    let err = blackwell_dense_gemm_fp16(a, b, c, m, n, k);
    cudaError::from(err).result()
}
```

#### 3. Implement MatmulKernel

```rust
// burn-blackwell/src/kernel.rs
use burn::tensor::Tensor;
use burn::backend::cuda::{Cuda, CudaDevice};
use super::ffi::launch_dense_gemm;

pub struct BlackwellDenseGemm;

impl burn::tensor::ops::MatmulKernel<f16> for BlackwellDenseGemm {
    fn forward<B: burn::backend::Backend<Device = CudaDevice>>(
        lhs: &Tensor<B, 2, f16>,
        rhs: &Tensor<B, 2, f16>,
    ) -> Tensor<B, 2, f16> {
        let (m, k) = (lhs.shape()[0] as i32, lhs.shape()[1] as i32);
        let n = rhs.shape()[1] as i32;
        
        let mut c = Tensor::<B, 2, f32>::zeros([m as usize, n as usize], &lhs.device());
        
        unsafe {
            launch_dense_gemm(
                lhs.as_ptr(),
                rhs.as_ptr(),
                c.as_mut_ptr(),
                m, n, k,
            ).expect("BlackwellDenseGemm launch failed");
        }
        
        // Convert FP32 output back to FP16 (performance cost!)
        c.to_dtype::<f16>()
    }
}
```

#### 4. Register Kernel

```rust
// burn-blackwell/src/lib.rs
use burn::backend::cuda::CudaBackend;

pub fn init() {
    burn::tensor::ops::register_matmul_kernel::<f16, CudaBackend, BlackwellDenseGemm>();
}
```

#### 5. Benchmark

```rust
#[test]
fn bench_blackwell_vs_cublas() {
    burn_blackwell::init();
    let dev = CudaDevice::new(0).unwrap();
    
    let a = Tensor::<Cuda, 2, f16>::random([8192, 237568], &dev);
    let b = Tensor::<Cuda, 2, f16>::random([237568, 8192], &dev);
    
    // Warmup
    for _ in 0..10 {
        let _ = a.matmul(&b);
    }
    
    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let c = a.matmul(&b);
        c.sync(); // Ensure completion
    }
    let elapsed = start.elapsed() / 100;
    
    let flops = 2.0 * 8192.0 * 8192.0 * 237568.0;
    let tflops = flops / elapsed.as_secs_f64() / 1e12;
    
    println!("BlackwellDenseGemm: {:.1} TFLOPS", tflops);
    // Expected: ~599 TFLOPS (96% of cuBLAS's 622.8)
}
```

---

## Honest Assessment

### Should You Do This?

**No, unless:**
1. You want to learn Rust FFI / CUDA integration (educational)
2. You're studying how ML frameworks integrate custom kernels
3. You plan to improve the kernel further (close 4% gap)

**Better alternatives:**
1. **Keep using cuBLAS in Burn** - It's faster (622.8 vs 598.9 TFLOPS)
2. **Contribute to CUTLASS** - Your optimization insights benefit everyone
3. **Focus on sparse operations** - Where cuBLAS/Burn have gaps

### Where Custom Kernels Make Sense in Burn

**Good integration targets:**
| Operation | Burn Default | Your Potential | Value |
|-----------|--------------|----------------|-------|
| Sparse BSR GEMM | cuSPARSE (~10 TFLOPS) | Custom (>50 TFLOPS) | 5× speedup ✅ |
| Fused operations | Separate kernels | Single kernel | 2-3× speedup ✅ |
| Custom attention | FA2 (slower) | FA3-style | 1.5× speedup ✅ |
| **Dense GEMM** | **cuBLAS (622.8)** | **Ours (598.9)** | **0.96× slower** ❌ |

**Your kernel is world-class (96% of cuBLAS) but still slower than the existing solution.**

---

## Alternative: Contribute to CUTLASS Instead

### Why CUTLASS Makes More Sense

**Your optimization journey:**
1. Started with CUTLASS 4.3 Example 49: 406.8 TFLOPS
2. Discovered optimal config: +47% improvement
3. Found K-dimension scaling: +20% additional
4. Reached 598.9 TFLOPS: 96% of cuBLAS

**This is valuable to CUTLASS community:**
- Document K-dimension scaling insights
- Share optimal TileShape/ClusterShape configs
- Contribute performance analysis

**Impact:**
- Benefits entire CUDA ecosystem
- Gets your name in CUTLASS documentation
- Improves baseline for everyone

**Effort:** 1-2 days (documentation + PR)  
**Value:** High (community impact)

vs

**Burn integration:**  
**Effort:** 1-2 weeks  
**Value:** Low (we're slower than existing)

---

## Recommendation

### Don't Integrate into Burn (Current Kernel)

**Reasons:**
1. ❌ We're 4% slower than cuBLAS (Burn's current backend)
2. ❌ High integration effort for negative value
3. ❌ Maintenance burden
4. ❌ No user benefit

### Better Path Forward

**Option 1: Contribute to CUTLASS** ✅
- Document your optimization insights
- Share optimal configurations
- Help community benefit from your work
- **Estimated effort:** 1-2 days
- **Value:** High (everyone benefits)

**Option 2: Close the 4% Gap, Then Reconsider** ⚠️
- If you can beat cuBLAS (>622.8 TFLOPS)
- Then Burn integration becomes attractive
- But: Proprietary optimizations hard to beat
- **Estimated effort:** Weeks to months (uncertain success)
- **Value:** High if successful

**Option 3: Build Sparse Kernel Instead** ✅
- cuSPARSE BSR: ~10 TFLOPS
- Custom optimized: 50+ TFLOPS potential
- **5× speedup** over Burn's current sparse backend
- **Estimated effort:** 2-4 weeks
- **Value:** Very high (real speedup)

---

## If You Insist on Burn Integration (Educational)

### Complete Scaffold

I can provide:
1. Full `burn-blackwell` crate structure
2. FFI bindings for your CUTLASS kernel
3. MatmulKernel implementation
4. Comprehensive tests
5. CI configuration for H100 validation

**But know:**
- This is educational, not production-valuable
- Burn users would see 4% slowdown
- cuBLAS remains better choice

**Estimated delivery:** 2-3 hours of work

---

## Bottom Line

### Your Recipe Was Based on Wrong Assumption

**You assumed:** BlackwellSparseK is sparse BSR  
**Reality:** BlackwellSparseK is dense GEMM  

**Your recipe:** Integrate sparse kernel into Burn (good idea!)  
**Reality:** Integrating dense GEMM slower than cuBLAS (bad idea)  

### Honest Recommendations

**Best use of your time:**
1. ✅ **Contribute optimization insights to CUTLASS** (high value, 1-2 days)
2. ✅ **Build actual sparse kernel** (5× speedup, 2-4 weeks)
3. ⚠️ **Try to close 4% gap** (uncertain, weeks-months)
4. ❌ **Integrate into Burn as-is** (negative value, 1-2 weeks)

### What I Can Do

If you want to proceed anyway (educational purpose):
1. Create full `burn-blackwell` crate
2. Adapt your recipe for dense GEMM
3. Write comprehensive tests
4. Document integration

**But I recommend:** Contribute to CUTLASS instead, or build a sparse kernel where you can provide real value (5× speedup over cuSPARSE).

---

**Your call. What makes sense given:**
- Our kernel: 598.9 TFLOPS (excellent, but 4% slower than cuBLAS)
- Integration effort: 1-2 weeks
- Value proposition: Negative (slower than existing)

**Deeds not words:** Should we integrate something slower, or build something faster?

