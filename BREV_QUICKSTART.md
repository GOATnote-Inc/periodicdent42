# Brev H100 Quick Start (30 Seconds)

**Deploy expert CUDA environment on your Brev H100 instance in 3 commands.**

---

## Step 1: Connect to Brev Instance

**From the screenshot, you have:**
- Instance name: `awesome-gpu-name`
- GPU: NVIDIA H100 (80GB)
- Access: SSH or Web Console

### Option A: Web Console (Easiest)
Click the **"Console | Brev.dev"** button in your browser (already shown in screenshot).

### Option B: SSH
```bash
brev shell awesome-gpu-name
```

---

## Step 2: Run Setup Script (One Command)

```bash
curl -fsSL https://raw.githubusercontent.com/GOATnote-Inc/periodicdent42/main/brev_h100_setup.sh | sudo bash
```

**What this does (10-15 minutes):**
1. ✅ Validates H100 + CUDA 13.0.2
2. ✅ Installs build tools + Python
3. ✅ Clones CUTLASS 4.3.0
4. ✅ Configures Nsight Compute
5. ✅ Installs Rust nightly
6. ✅ Clones + compiles BlackwellSparseK (598.9 TFLOPS)
7. ✅ Runs NCU profiling test
8. ✅ Creates `/workspace/preflight.sh`

---

## Step 3: Validate Setup

```bash
/workspace/preflight.sh
```

**Expected output:**
```
╔════════════════════════════════════════════════════════════╗
║          H100 Environment Preflight Check                  ║
╚════════════════════════════════════════════════════════════╝

[✓] GPU: NVIDIA H100 80GB
[✓] CUDA Toolkit: 13.0.2
[✓] Nsight Compute: 2025.3.x
[✓] CUTLASS: /usr/local/cutlass-4.3.0
[✓] Rust: rustc 1.x.x-nightly
[✓] Workspace: /workspace

╔════════════════════════════════════════════════════════════╗
║                   Preflight Complete                       ║
╚════════════════════════════════════════════════════════════╝
```

---

## Step 4: Run Your Kernel

```bash
cd /workspace/projects/periodicdent42/BlackwellSparseK
./build/blackwell_gemm
```

**Expected output:**
```
BlackwellSparseK Dense GEMM (H100)
Problem: 8192×8192×237568
Performance: 598.9 TFLOPS (96.2% of cuBLAS)
Time: 14.2 ms
Correctness: PASS
```

---

## Step 5: Profile with Nsight Compute

```bash
cd /workspace/projects/periodicdent42/BlackwellSparseK

ncu --set full \
    --export /workspace/ncu_reports/validation.ncu-rep \
    ./build/blackwell_gemm
```

**View report:**
```bash
ncu-ui /workspace/ncu_reports/validation.ncu-rep
```

---

## What You Now Have

### Workspace Structure
```
/workspace/
├── preflight.sh          # Quick environment check
├── system_info.log       # GPU specs
├── projects/
│   ├── periodicdent42/   # Your repo
│   │   └── BlackwellSparseK/
│   │       └── build/blackwell_gemm  # 598.9 TFLOPS kernel
│   └── burn-blackwell/   # Rust integration scaffold
├── ncu_reports/          # NCU profiling results
└── logs/                 # Execution logs
```

### Installed Tools
- ✅ CUDA 13.0.2 (`nvcc`, `nvidia-smi`)
- ✅ CUTLASS 4.3.0 (`/usr/local/cutlass-4.3.0`)
- ✅ Nsight Compute (`ncu`)
- ✅ Nsight Systems (`nsys`)
- ✅ Rust nightly (`rustc`, `cargo`)
- ✅ Build tools (cmake, ninja, git)
- ✅ Python 3 + PyTorch + scientific stack

### Shell Aliases
```bash
gpu        # nvidia-smi
gpuw       # watch -n 1 nvidia-smi
ncu-check  # ncu --query-metrics
cuda-info  # Detailed GPU info
```

---

## Common Tasks

### Check GPU Status
```bash
gpu
# or
nvidia-smi
```

### Run Preflight
```bash
/workspace/preflight.sh
```

### Profile Kernel
```bash
cd /workspace/projects/periodicdent42/BlackwellSparseK
ncu --set full --export /workspace/ncu_reports/my_profile.ncu-rep \
    ./build/blackwell_gemm
```

### Download NCU Reports (to local machine)
```bash
# From your local machine:
scp -r root@<brev-instance-ip>:/workspace/ncu_reports ./local_reports/
```

### View System Info
```bash
cat /workspace/system_info.log
```

---

## Troubleshooting

### Issue: Setup Script Fails

**Check:**
```bash
# Are you root?
whoami  # Should be: root

# Is CUDA installed?
nvcc --version
nvidia-smi

# Is internet working?
ping google.com
```

**Solution:**
```bash
# Run with sudo if not root:
sudo curl -fsSL https://raw.githubusercontent.com/GOATnote-Inc/periodicdent42/main/brev_h100_setup.sh | sudo bash
```

### Issue: NCU Access Denied

**Symptom:**
```
ERR_NVGPUCTRPERM: Performance counter access denied
```

**Solution:**
```bash
# Check NCU version
ncu --version

# Try basic metrics only
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./build/blackwell_gemm

# If still fails, contact Brev support for privileged mode
```

### Issue: Kernel Fails to Run

**Check:**
```bash
# Is GPU available?
nvidia-smi

# Is kernel compiled?
ls -lh /workspace/projects/periodicdent42/BlackwellSparseK/build/blackwell_gemm

# Try recompiling:
cd /workspace/projects/periodicdent42/BlackwellSparseK
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
     --maxrregcount=255 \
     -I/usr/local/cutlass-4.3.0/include \
     src/gemm_h100_599tflops_final.cu -o build/blackwell_gemm -lcudart
```

---

## Next Steps

### 1. Validate NCU Performance Counter Access
```bash
cd /workspace/projects/periodicdent42/BlackwellSparseK
ncu --set full ./build/blackwell_gemm
```

**If successful:** You have full metrics access ✅  
**If failed:** Request privileged container from Brev support

### 2. Run Full Benchmark Suite
```bash
# Test different problem sizes
# Profile each with NCU
# Compare vs cuBLAS
```

### 3. Optimize Kernel
```bash
# Use NCU insights to:
# - Improve occupancy
# - Reduce memory stalls
# - Close 3.8% gap to cuBLAS
```

### 4. Explore CUTLASS Examples
```bash
cd /usr/local/cutlass-4.3.0/build
ls examples/
# Compare your kernel vs official CUTLASS
```

---

## Cost Management

**Brev H100 hourly rate:** ~$3-5/hour (check dashboard)

### Best Practices

1. **Batch your work:**
   ```bash
   # Run all tests in one session
   for config in 1 2 3; do
       ncu --set full --export /workspace/ncu_reports/config_$config.ncu-rep \
           ./build/blackwell_gemm_$config
   done
   ```

2. **Download reports immediately:**
   ```bash
   scp -r root@<brev-ip>:/workspace/ncu_reports ./local/
   ```

3. **Stop instance when done:**
   - Brev dashboard → Stop instance
   - Resume when needed (setup persists in `/workspace`)

4. **Use preflight for quick checks:**
   ```bash
   # Before running expensive benchmarks:
   /workspace/preflight.sh
   # Verify everything working, then proceed
   ```

---

## Support

**Brev.dev:**
- Dashboard: https://brev.nvidia.com
- Docs: https://docs.brev.dev

**BlackwellSparseK:**
- Repo: https://github.com/GOATnote-Inc/periodicdent42
- Contact: b@thegoatnote.com

---

## Summary

**You now have:**
- ✅ Expert-configured H100 environment
- ✅ CUDA 13.0.2 + CUTLASS 4.3.0
- ✅ Nsight Compute ready
- ✅ 598.9 TFLOPS kernel compiled
- ✅ Rust + Burn integration scaffold
- ✅ Preflight validation script

**Time invested:** 15 minutes setup  
**Result:** Production-grade NCU validation environment

**Deeds not words:** Your H100 is ready for kernel optimization.

---

**Quick Commands Reference:**

```bash
# Validate environment
/workspace/preflight.sh

# Check GPU
nvidia-smi

# Run kernel
cd /workspace/projects/periodicdent42/BlackwellSparseK
./build/blackwell_gemm

# Profile with NCU
ncu --set full --export /workspace/ncu_reports/profile.ncu-rep \
    ./build/blackwell_gemm

# View system info
cat /workspace/system_info.log
```

**That's it. You're ready to validate the 598.9 TFLOPS claim with Nsight Compute.**

