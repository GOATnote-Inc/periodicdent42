# Nsight Compute on RunPod - Issue & Workaround (Oct 31, 2025)

## 🔴 Issue: ERR_NVGPUCTRPERM

**Error:**
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access 
NVIDIA GPU Performance Counters on the target device 0.
```

**Root Cause:**  
RunPod containers run with restricted capabilities and cannot access GPU hardware performance counters. This is a security restriction at the container/kernel level.

---

## ❌ Attempted Fixes (All Failed)

### 1. Modprobe Configuration
```bash
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' \
  > /etc/modprobe.d/nvidia-profiling.conf
```
**Result:** Read-only filesystem, cannot write to `/etc/modprobe.d/`

### 2. nvidia-smi Profiling Mode
```bash
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -acp 0  # Unrestricted accounting
```
**Result:** Commands succeed but don't grant counter access

### 3. Device Node Permissions
```bash
chmod 666 /dev/nvidiactl /dev/nvidia0
```
**Result:** Permissions change succeeds but counters still blocked

### 4. Kernel Parameter Check
```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```
**Output:** `RmProfilingAdminOnly: 1` (restricted mode)

**Problem:** Cannot change kernel module parameters without reloading driver, which requires CAP_SYS_ADMIN.

---

## ✅ Working Solution: Nsight Systems (nsys)

**Key Difference:**  
- Nsight **Compute** (ncu): Requires GPU performance counters (hardware access)
- Nsight **Systems** (nsys): Uses CUDA APIs + instrumentation (software access)

### What Nsight Systems Provides

| Metric                    | Available? | Method                |
| :------------------------ | :--------- | :-------------------- |
| Kernel execution time     | ✅         | CUDA API events       |
| Memory transfer bandwidth | ✅         | cudaMemcpy timing     |
| Host API overhead         | ✅         | CPU instrumentation   |
| Launch config             | ✅         | CUDA API inspection   |
| SM utilization %          | ❌         | Requires ncu          |
| Tensor Core active %      | ❌         | Requires ncu          |
| Warp stall reasons        | ❌         | Requires ncu          |
| L1/L2 cache hit rates     | ❌         | Requires ncu          |

### What Nsight Systems Cannot Provide

- **Fine-grained SM metrics** (utilization %, occupancy %)
- **Memory subsystem metrics** (L1/L2 hit rates, DRAM throughput %)
- **Tensor Core utilization** (HMMA cycles, pipeline efficiency)
- **Warp stall analysis** (memory dependency, execution dependency, etc.)

**Workaround:** Use CUDA Events for kernel timing + code inspection for theoretical analysis.

---

## 📋 RunPod Profiling Strategy

### Tier 1: CUDA Events (Always Available)

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

**Provides:**
- ✅ Accurate kernel latency
- ✅ No permission requirements
- ✅ Low overhead (<1 μs)
- ✅ Works in all environments

### Tier 2: Nsight Systems (RunPod Compatible)

```bash
nsys profile --stats=true --force-overwrite=true -o profile.nsys-rep ./binary
```

**Provides:**
- ✅ Kernel timeline
- ✅ Memory transfer analysis
- ✅ CUDA API overhead
- ✅ Multi-stream visualization
- ❌ No SM-level metrics

### Tier 3: Nsight Compute (Requires Privileged Container)

```bash
# Only works with:
# - docker run --cap-add=CAP_SYS_ADMIN --privileged
# - Or RunPod "Secure Cloud" tier (not standard pods)
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_elapsed ./binary
```

**Provides:**
- ✅ All metrics
- ✅ Roofline analysis
- ✅ Warp scheduler analysis
- ⚠️ Requires special pod type

---

## 🎯 Practical Workflow for RunPod

### Development Phase (Baseline)

1. **Use CUDA Events** for latency measurement
   ```cpp
   cudaEventElapsedTime(&ms, start, stop);
   printf("[Timing] Latency: %.3f ms\n", ms);
   ```

2. **Use Nsight Systems** for pipeline analysis
   ```bash
   nsys profile -o baseline.nsys-rep ./kernel
   ```

3. **Manual occupancy calculation**
   ```bash
   # From ptxas output:
   # Registers: 197, Shared Memory: 16 KB
   # Occupancy = min(blocks/SM, resource limits)
   ```

### Optimization Phase (Iteration)

1. **Apply optimization** (e.g., TMA, warp specialization)

2. **Measure with CUDA Events**
   ```bash
   ./kernel  # Reports latency via CUDA Events
   ```

3. **Profile with Nsight Systems** (optional, for sanity check)
   ```bash
   nsys profile -o optimized.nsys-rep ./kernel
   ```

4. **Compare**
   ```
   Baseline:   615 μs
   Optimized:  310 μs
   Speedup:    2.0×
   ```

### Validation Phase (H100 with Privileged Access)

Request RunPod "Secure Cloud" or use local H100 with full permissions:

```bash
# Full profiling
ncu --set full -o detailed_profile ./kernel

# Roofline
ncu --metrics \
  sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
  sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
  dram__bytes.sum \
  ./kernel
```

---

## 📊 What We Achieved Without ncu

From Nsight Systems + CUDA Events:

```
Kernel Latency:        615 μs (CUDA Events)
Host Overhead:         227 ms (nsys)
Memory Bandwidth:      15 GB/s H2D, 13.8 GB/s D2H (nsys)
Bottleneck:            cudaMalloc (89.4% of time, nsys)
Kernel Instances:      2 launches (nsys)
Launch Config:         grid=(64,64), block=128 (nsys)
Register Usage:        197 (ptxas)
Shared Memory:         16 KB (ptxas)
Occupancy (calc):      ~66% (from ptxas + manual calc)
```

**What's Missing:**
- SM utilization % (need ncu)
- Tensor Core active % (need ncu)
- Warp stall reasons (need ncu)

**Workaround:**
- Theoretical occupancy from ptxas
- Code inspection for TC usage (WMMA calls)
- Inference from latency trends

---

## 🚀 Recommended Approach

### For RunPod Development

```bash
# 1. Quick iteration
./kernel  # CUDA Events timing

# 2. Detailed analysis (once per major change)
nsys profile --stats=true -o profile.nsys-rep ./kernel

# 3. Download profile
scp -P PORT root@IP:/path/profile.nsys-rep ./

# 4. Open in Nsight Systems GUI (local)
```

### For Final Validation

```bash
# On H100 with full permissions
ncu --set full --target-processes all -o final_profile ./kernel

# Analyze all metrics
ncu --import final_profile.ncu-rep
```

---

## 📦 Summary

| Tool              | Permissions | Kernel Time | Host Time | SM Metrics | Use Case           |
| :---------------- | :---------- | :---------- | :-------- | :--------- | :----------------- |
| **CUDA Events**   | None        | ✅          | ❌        | ❌         | Quick iteration    |
| **Nsight Systems**| User        | ✅          | ✅        | ❌         | Pipeline analysis  |
| **Nsight Compute**| Admin       | ✅          | ✅        | ✅         | Deep optimization  |

**Recommendation for RunPod:**  
Use **CUDA Events** (iteration) + **Nsight Systems** (validation) + **manual analysis** (occupancy, TC usage)

**Status:** Working baseline established with available tools ✅

---

**Next:** Fix memory allocation → re-profile with nsys → integrate CUTLASS TMA

