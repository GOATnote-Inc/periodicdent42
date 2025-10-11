# 🖥️ GPU Setup Guide: Phase 2-5 Execution

**Target Hardware**: T4 → A100 → H100  
**Strategy**: Smart GPU tiering for cost efficiency  
**Total Budget**: $89-165 (85% under $1,000 budget)  

---

## 📋 Overview

This guide provides step-by-step instructions for setting up GPU instances on Google Cloud Platform (GCP) to validate and benchmark CUDAdent42 kernels.

**Philosophy**: *Code like a researcher, spend like an engineer.*

**Strategy**:
- **Phase 2 (T4)**: Initial validation - $0.11/hr preemptible
- **Phase 3 (A100)**: Optimization - $1.10/hr preemptible
- **Phase 4 (H100)**: Hopper features - $3.67/hr on-demand
- **Phase 5 (H100)**: Final benchmarks - $3.67/hr on-demand

**Key Principles**:
1. Use cheapest GPU that validates each phase
2. Aggressive start/stop management
3. Batch all GPU work for efficiency
4. Save H100 for final showcase only

---

## 🚀 Phase 2: T4 GPU Validation ($5-10 budget)

**Goal**: Verify kernel compiles and runs correctly  
**Time**: 30-50 hours GPU time (spread over 2 days real-time)  
**Cost**: ~$5-10 total

### Step 1: Create T4 Instance

```bash
# Set project and zone
gcloud config set project periodicdent42
gcloud config set compute/zone us-central1-a

# Create preemptible T4 instance
gcloud compute instances create cudadent42-t4-dev \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --preemptible \
    --boot-disk-size=100GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform

# SSH into instance
gcloud compute ssh cudadent42-t4-dev --zone=us-central1-a
```

### Step 2: Setup Auto-Shutdown (Save Money!)

```bash
# Create auto-shutdown script (stops if idle 10+ minutes)
cat > /home/$USER/auto_shutdown.sh << 'EOF'
#!/bin/bash
# Auto-shutdown if no GPU activity for 10 minutes

IDLE_TIME=600  # 10 minutes
CHECK_INTERVAL=60  # 1 minute

while true; do
    # Check if any process is using GPU
    if ! nvidia-smi | grep -q 'python\|pytorch'; then
        # No GPU activity, wait IDLE_TIME then shutdown
        sleep $IDLE_TIME
        
        # Check again (in case activity started)
        if ! nvidia-smi | grep -q 'python\|pytorch'; then
            echo "No GPU activity detected. Shutting down to save costs..."
            sudo poweroff
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
EOF

chmod +x /home/$USER/auto_shutdown.sh

# Run in background
nohup /home/$USER/auto_shutdown.sh > /home/$USER/auto_shutdown.log 2>&1 &
```

### Step 3: Clone Repository and Setup

```bash
# Clone repository
cd ~
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42/cudadent42

# Create conda environment
conda create -n flashmoe python=3.12 -y
conda activate flashmoe

# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 4: Build CUDA Extension

```bash
# Build
python setup.py build_ext --inplace

# If build succeeds, run tests
pytest tests/test_warp_specialized.py -v --tb=short

# Expected: Some tests may pass, some may fail (that's OK for Phase 2)
# Goal: Verify compilation and basic functionality
```

### Step 5: Fix Issues Locally (Stop Instance!)

```bash
# Copy test results to Cloud Storage
gsutil cp -r test-results gs://my-bucket/cudadent42/phase2/

# STOP INSTANCE to save money
gcloud compute instances stop cudadent42-t4-dev --zone=us-central1-a

# Analyze locally, fix code, push to GitHub
# Restart instance and iterate
```

---

## 🔧 Phase 3: A100 Optimization ($55-100 budget)

**Goal**: Profile and optimize for performance  
**Time**: 50-90 hours GPU time (1 week real-time, multiple sessions)  
**Cost**: ~$55-100 total

### Step 1: Create A100 Instance (Preemptible!)

```bash
# Create preemptible A100 instance (70% cheaper than on-demand)
gcloud compute instances create cudadent42-a100-opt \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --preemptible \
    --boot-disk-size=200GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform

# SSH into instance
gcloud compute ssh cudadent42-a100-opt --zone=us-central1-a
```

### Step 2: Create Disk Snapshot (Fast Resume)

```bash
# After initial setup, create snapshot for fast resume
gcloud compute disks snapshot cudadent42-a100-opt \
    --snapshot-names=cudadent42-a100-base \
    --zone=us-central1-a

# To restore from snapshot (future sessions):
gcloud compute disks create cudadent42-a100-opt-new \
    --source-snapshot=cudadent42-a100-base \
    --zone=us-central1-a
```

### Step 3: Install Nsight Compute

```bash
# Download Nsight Compute (profiling tool)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-compute-2024.1.1_2024.1.1.1-1_amd64.deb
sudo dpkg -i nsight-compute-2024.1.1_2024.1.1.1-1_amd64.deb

# Verify
ncu --version
```

### Step 4: Optimization Workflow (1-2 hour sessions)

```bash
# Session 1: Profile
ncu --set full --export profile_iter1 python benchmarks/benchmark_attention.py
gsutil cp profile_iter1.ncu-rep gs://my-bucket/cudadent42/phase3/
gcloud compute instances stop cudadent42-a100-opt --zone=us-central1-a

# Analyze locally with Nsight Compute UI
# Identify optimization opportunities

# Session 2: Fix issues, re-profile
gcloud compute instances start cudadent42-a100-opt --zone=us-central1-a
# ... test optimizations ...
gcloud compute instances stop cudadent42-a100-opt --zone=us-central1-a

# Repeat until performance targets met
```

**Target Metrics** (from Nsight):
- SM Occupancy: ≥85%
- Memory Bandwidth: ≥80% peak
- Warp Efficiency: ≥95%

---

## 🏆 Phase 4: H100 Hopper Features ($18-37 budget)

**Goal**: Add H100-specific optimizations  
**Time**: 5-10 hours GPU time (2-3 focused sessions)  
**Cost**: ~$18-37 total

### Step 1: Create H100 Instance (On-Demand)

```bash
# H100 only available on-demand (not preemptible)
gcloud compute instances create cudadent42-h100-dev \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-1g \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --boot-disk-size=100GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform

# SSH into instance
gcloud compute ssh cudadent42-h100-dev --zone=us-central1-a
```

### Step 2: Verify Hopper Architecture

```bash
# Check SM version (should be sm_90 for H100)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Should output: 9.0 (Hopper)
```

### Step 3: Test Hopper Features

```bash
# Compile with Hopper optimizations
export CUDA_ARCH="90"
python setup.py build_ext --inplace

# Run tests
pytest tests/test_warp_specialized.py -v

# Benchmark
python benchmarks/benchmark_attention.py --save-results --output-dir results/h100/

# IMMEDIATELY stop after completion
gcloud compute instances stop cudadent42-h100-dev --zone=us-central1-a
```

---

## 📊 Phase 5: Final Benchmarks ($11-18 budget)

**Goal**: Comprehensive benchmarks for publication  
**Time**: 3-5 hours GPU time (single automated session)  
**Cost**: ~$11-18 total

### Step 1: Create Automated Benchmark Script

```bash
# Create comprehensive benchmark script
cat > /home/$USER/run_final_benchmarks.sh << 'EOF'
#!/bin/bash
set -e

# Activate environment
conda activate flashmoe

# Run comprehensive benchmarks
python benchmarks/benchmark_attention.py \
    --seq-lens 512 1024 2048 4096 8192 \
    --batch-size 4 \
    --num-heads 8 \
    --save-results \
    --output-dir /home/$USER/results/final/

# Profile key configurations with Nsight
ncu --set full --export /home/$USER/results/final/profile_2048 \
    --force-overwrite \
    python -c "
import torch
import flashmoe_science_ext

Q = torch.randn(4, 8, 2048, 64, dtype=torch.bfloat16, device='cuda')
K = torch.randn(4, 8, 2048, 64, dtype=torch.bfloat16, device='cuda')
V = torch.randn(4, 8, 2048, 64, dtype=torch.bfloat16, device='cuda')

O = flashmoe_science_ext.flash_attention_warp_specialized(Q, K, V, causal=False, softmax_scale=0.125)
"

# Upload results to Cloud Storage
gsutil -m cp -r /home/$USER/results/final/* gs://my-bucket/cudadent42/phase5/

# Auto-shutdown after completion
echo "Benchmarks complete! Shutting down..."
sudo poweroff
EOF

chmod +x /home/$USER/run_final_benchmarks.sh
```

### Step 2: Run Automated Benchmarks

```bash
# Start H100 instance
gcloud compute instances start cudadent42-h100-dev --zone=us-central1-a

# SSH and run benchmarks
gcloud compute ssh cudadent42-h100-dev --zone=us-central1-a

# Run automated script
nohup /home/$USER/run_final_benchmarks.sh > /home/$USER/benchmark.log 2>&1 &

# Detach (instance will auto-shutdown when done)
exit
```

### Step 3: Retrieve Results

```bash
# Download results from Cloud Storage
gsutil -m cp -r gs://my-bucket/cudadent42/phase5/ ./results/

# Generate publication-quality graphs locally
python benchmarks/benchmark_attention.py --plot-only --input-dir results/phase5/
```

---

## 💰 Cost Tracking

### GCP Billing Commands

```bash
# Check current month spending
gcloud billing accounts list
gcloud billing projects describe periodicdent42

# Set budget alerts (recommended)
gcloud billing budgets create \
    --billing-account=YOUR_BILLING_ACCOUNT_ID \
    --display-name="CUDAdent42 GPU Budget" \
    --budget-amount=200 \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=75 \
    --threshold-rule=percent=90
```

### Cost Estimation

| Phase | GPU | Hours | Rate/hr | Total |
|-------|-----|-------|---------|-------|
| Phase 2 | T4 (preempt) | 30-50 | $0.11 | $5-10 |
| Phase 3 | A100 (preempt) | 50-90 | $1.10 | $55-100 |
| Phase 4 | H100 (on-demand) | 5-10 | $3.67 | $18-37 |
| Phase 5 | H100 (on-demand) | 3-5 | $3.67 | $11-18 |
| **TOTAL** | | | | **$89-165** |

**Safety Buffer**: $835-911 remaining (83-91% under budget)

---

## 🔧 Troubleshooting

### Issue: Instance Preempted (A100)

```bash
# Preemptible instances can be terminated by GCP
# Solution: Create snapshot frequently

# If preempted, recreate from snapshot:
gcloud compute instances create cudadent42-a100-opt-new \
    --source-instance-template=cudadent42-a100-template \
    --boot-disk-snapshot=cudadent42-a100-base
```

### Issue: CUDA Out of Memory

```bash
# Reduce batch size or sequence length
python benchmarks/benchmark_attention.py --batch-size 2 --seq-lens 512 1024

# Check GPU memory
nvidia-smi
```

### Issue: Compilation Errors

```bash
# Check CUDA version
nvcc --version

# Rebuild with verbose output
python setup.py build_ext --inplace --verbose

# Check compiler flags in setup.py
```

### Issue: Slow Performance

```bash
# Profile with Nsight to identify bottlenecks
ncu --set full python benchmarks/benchmark_attention.py

# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active

# Check memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed
```

---

## ✅ Success Criteria

Before moving to next phase, verify:

**Phase 2 (T4)**:
- ✅ Code compiles without errors
- ✅ Tests run (some passes expected)
- ✅ No memory errors or crashes
- ✅ Numerical correctness within tolerance

**Phase 3 (A100)**:
- ✅ All tests pass
- ✅ Performance ≥1.5x PyTorch SDPA
- ✅ SM Occupancy ≥70%
- ✅ No obvious optimization opportunities

**Phase 4 (H100)**:
- ✅ Hopper features compile and run
- ✅ Performance ≥1.8x PyTorch SDPA
- ✅ SM Occupancy ≥85%
- ✅ Warp efficiency ≥90%

**Phase 5 (H100)**:
- ✅ Comprehensive benchmarks complete
- ✅ Performance graphs generated
- ✅ Nsight profiles captured
- ✅ Results published to Cloud Storage

---

## 📚 Additional Resources

**GCP Documentation**:
- [GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [Deep Learning VM Images](https://cloud.google.com/deep-learning-vm)
- [Preemptible VMs](https://cloud.google.com/compute/docs/instances/preemptible)

**NVIDIA Documentation**:
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)

**Performance Optimization**:
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)

---

## 🎯 Quick Reference Commands

```bash
# Start instance
gcloud compute instances start INSTANCE_NAME --zone=us-central1-a

# Stop instance (SAVE MONEY!)
gcloud compute instances stop INSTANCE_NAME --zone=us-central1-a

# SSH
gcloud compute ssh INSTANCE_NAME --zone=us-central1-a

# Copy files to Cloud Storage
gsutil cp -r results/ gs://my-bucket/cudadent42/

# Check GPU status
nvidia-smi

# Monitor costs
gcloud billing projects describe periodicdent42
```

---

**End of GPU Setup Guide**

*This guide follows the principle: "Code like a researcher, spend like an engineer."*

**Ready for Phase 2 GPU validation!** 🚀

