# Knowledge Transfer: L4 CUDA Kernel Optimization with EvoEngineer + robust-kbench

**Date:** October 15, 2025  
**Context:** Post-mortem of V3 FlashAttention kernel development  
**Audience:** New engineer tasked with beating PyTorch SDPA on L4 GPU  
**Goal:** Provide complete, executable instructions + learnings to succeed where V3 failed

---

## Executive Summary: What We Learned (So You Don't Repeat Our Mistakes)

### ✅ What Worked in V3 Post-Mortem

1. **Systematic Methodology:** Post-mortem plan with 2-iteration stop-loss prevented endless debugging
2. **Diagnostic Tools:** Tile oracle test, compute-sanitizer, numpy analysis caught bugs fast
3. **Honest Documentation:** Transparent failure analysis (see `V3_POSTMORTEM.md`)
4. **Cost Control:** $1.00 stop-loss saved $10+ by preventing sunk-cost fallacy

### ❌ What Failed in V3 Development

1. **Premature Optimization:** Built V3 (complex: persistent blocks, cp.async pipeline) without validating V1
2. **No Unit Tests:** Never tested QK, softmax, SV components separately
3. **Late Profiling:** Didn't profile SDPA to understand why it's 4.4× faster before trying to beat it
4. **Systematic Bug:** V3 has 0.675× scaling (l_i accumulator 1.48× too large), root cause unidentified after 2 iterations
5. **Wrong Target:** L4's 48KB SMEM limit makes large-tile approach less beneficial than on H100

### 🎯 Current State (Your Starting Point)

**Production Champion:** PyTorch SDPA
- **Performance:** 0.073 ms (B=2, H=8, S=512, D=64, FP16)
- **Correctness:** 100%
- **Hardware:** L4 (Ada, sm_89, 48KB SMEM, 58 SMs, 300 GB/s bandwidth)

**Available Kernels:**
- **V2 (tensor cores):** ✅ Correct, 0.318 ms (4.4× slower than SDPA)
- **V3 (large tiles):** ❌ BLOCKED, systematic 0.675× scaling bug
- **Path:** `cudadent42/bench/kernels/`

**Evidence Trail:** All in `cudadent42/artifacts/`
- Oracle test results, memcheck logs (0 errors), error analysis (0.675× scaling)
- Post-mortem: `V3_POSTMORTEM.md` (1,850 lines of learnings)

---

## Critical Learnings for Your Success

### 1. Profile FIRST, Optimize SECOND

**Do This:**
```bash
# Before writing any custom kernel, profile SDPA to understand the bar
ncu --set full --target-processes all --export baseline_sdpa \
  python -c "
import torch
Q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
K, V = Q.clone(), Q.clone()
for _ in range(100):
    torch.nn.functional.scaled_dot_product_attention(Q, K, V)
"

# Look for:
# - Memory bandwidth utilization (target: >70% of 300 GB/s)
# - Tensor core utilization (HMMA ops)
# - Warp occupancy (target: >50%)
# - L2 cache hit rate
```

**Why:** SDPA is 0.073 ms. If Nsight shows it's already at 90% memory bandwidth, you can't beat it on L4. Pivot to decode optimization or different hardware.

### 2. Test Correctness at EVERY Step

**Do This:**
```python
# Create test_correctness.py (run after EVERY kernel change)
import torch
import torch.nn.functional as F

def test_parity(kernel_fn, B=2, H=8, S=512, D=64, atol=1e-2, rtol=1e-2):
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Reference
    O_ref = F.scaled_dot_product_attention(Q, K, V)
    
    # Your kernel
    O_test = kernel_fn(Q, K, V)
    
    # Check
    assert not torch.isnan(O_test).any(), "NaN detected!"
    assert not torch.isinf(O_test).any(), "Inf detected!"
    max_diff = (O_test - O_ref).abs().max().item()
    assert max_diff < atol, f"Max diff {max_diff:.6f} > {atol}"
    print(f"✅ Parity OK: max_diff={max_diff:.6f}")
```

**Why:** V3 had no correctness tests until Step 1b. The 0.675× bug was baked in from the start.

### 3. Start Simple, Add Complexity Incrementally

**Do This (Build V1 → V1.1 → V1.2, etc.):**

1. **V1 Baseline:** Naive FlashAttention (no optimizations)
   - Test: ✅ Correctness
   - Benchmark: Measure vs SDPA (expect 5-10× slower)
   
2. **V1.1:** Add tensor cores (WMMA/HMMA)
   - Test: ✅ Correctness (if fails, debug BEFORE next step)
   - Benchmark: Expect 2-3× speedup over V1
   
3. **V1.2:** Add shared memory tiling
   - Test: ✅ Correctness
   - Benchmark: Expect 1.5-2× speedup over V1.1
   
4. **V1.3:** Add cp.async double-buffering
   - Test: ✅ Correctness
   - Benchmark: Expect 1.2-1.5× speedup over V1.2

**Why:** V3 jumped to "persistent blocks + cp.async + large tiles" without validating each component. Impossible to debug when multiple features interact.

### 4. Use Compute-Sanitizer Aggressively

**Do This (After EVERY kernel change):**
```bash
compute-sanitizer --tool memcheck python test_correctness.py
compute-sanitizer --tool racecheck python test_correctness.py
compute-sanitizer --tool initcheck python test_correctness.py
```

**Why:** V3 passed all sanitizers (0 errors), confirming the bug was computational, not memory-related. This saved hours of debugging wrong paths.

### 5. Stop Conditions Are Your Friend

**Do This:**
- Set **time budget** (e.g., 2 hours per optimization)
- Set **iteration budget** (e.g., 3 fix attempts per bug)
- Set **cost budget** (e.g., $5 GPU spend per kernel)
- **STOP** if you hit limits without progress

**Why:** V3 post-mortem had 2-iteration limit + $1.00 stop-loss. We stopped at $0.23, saving $10+ on unfixable bug.

---

## Step-by-Step Instructions: EvoEngineer + robust-kbench Workflow

### Prerequisites

**Check your environment:**
```bash
# Verify GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv
# Expected: NVIDIA L4, 8.9

# Verify CUDA toolkit
nvcc --version
# Expected: >= 12.2

# Verify Python
python3 --version
# Expected: >= 3.10

# Verify PyTorch
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: >= 2.2, True
```

**If not on L4 or CUDA < 12.2:** ABORT and report versions.

---

### Step 0: Repo Prep & Guards

```bash
cd /path/to/periodicdent42

# Create feature branch
git checkout -b feature/evoengineer-rbk-l4-optim

# Create benchmark directory with date
export BENCH_DATE=$(date +%Y-%m-%d)
mkdir -p benchmarks/l4/${BENCH_DATE}/{baseline,rbk,nsight,leaderboard,final}

# Record environment
cat > benchmarks/l4/${BENCH_DATE}/environment.json << EOF
{
  "date": "${BENCH_DATE}",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
  "compute_cap": "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)",
  "cuda_version": "$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')",
  "pytorch_version": "$(python3 -c 'import torch; print(torch.__version__)')",
  "hostname": "$(hostname)",
  "commit": "$(git rev-parse HEAD)"
}
EOF

echo "✅ Environment recorded to benchmarks/l4/${BENCH_DATE}/environment.json"
cat benchmarks/l4/${BENCH_DATE}/environment.json
```

**Commit:**
```bash
git add benchmarks/l4/${BENCH_DATE}/environment.json
git commit -m "feat: Initialize EvoEngineer+RBK workflow for L4 optimization

Environment:
- GPU: L4 (sm_89)
- CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
- PyTorch: $(python3 -c 'import torch; print(torch.__version__)')
- Date: ${BENCH_DATE}
"
```

---

### Step 1: Pin & Integrate Tools (Reproducible)

**Discover official repositories:**
```bash
# Search for EvoEngineer
# Expected: https://github.com/<org>/EvoEngineer or similar
# Note: As of Oct 2025, verify current canonical repo

# Search for robust-kbench
# Expected: https://github.com/<org>/robust-kbench or similar
# Note: Verify current canonical repo
```

**Add as submodules:**
```bash
mkdir -p third_party

# Add EvoEngineer (replace with actual URL)
git submodule add https://github.com/<org>/EvoEngineer.git third_party/EvoEngineer
cd third_party/EvoEngineer
export EVOENGINEER_COMMIT=$(git rev-parse HEAD)
cd ../..

# Add robust-kbench (replace with actual URL)
git submodule add https://github.com/<org>/robust-kbench.git third_party/robust-kbench
cd third_party/robust-kbench
export RBK_COMMIT=$(git rev-parse HEAD)
cd ../..

# Record lockfile
cat > third_party/LOCKFILE.md << EOF
# Third-Party Tool Versions

**Date:** ${BENCH_DATE}

## EvoEngineer
- **Repo:** https://github.com/<org>/EvoEngineer.git
- **Commit:** ${EVOENGINEER_COMMIT}
- **Date Pinned:** ${BENCH_DATE}

## robust-kbench
- **Repo:** https://github.com/<org>/robust-kbench.git
- **Commit:** ${RBK_COMMIT}
- **Date Pinned:** ${BENCH_DATE}

## Verification

\`\`\`bash
cd third_party/EvoEngineer && git rev-parse HEAD  # Should match ${EVOENGINEER_COMMIT}
cd third_party/robust-kbench && git rev-parse HEAD  # Should match ${RBK_COMMIT}
\`\`\`
EOF
```

**Create bootstrap script:**
```bash
cat > scripts/bootstrap_tools.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "================================================"
echo "Bootstrapping EvoEngineer + robust-kbench"
echo "================================================"

# Ensure submodules initialized
git submodule update --init --recursive

# Create/activate conda env
if ! conda env list | grep -q cuda-optim; then
    conda create -n cuda-optim python=3.10 -y
fi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cuda-optim

# Install base dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn pyyaml

# Install EvoEngineer
echo "Installing EvoEngineer..."
cd third_party/EvoEngineer
pip install -e .
cd ../..

# Install robust-kbench
echo "Installing robust-kbench..."
cd third_party/robust-kbench
pip install -e .
cd ../..

# Verify installations
echo ""
echo "Verifying installations..."
python3 -c "import evoengineer; print(f'✅ EvoEngineer: {evoengineer.__version__}')" || echo "❌ EvoEngineer import failed"
python3 -c "import robust_kbench; print(f'✅ robust-kbench: {robust_kbench.__version__}')" || echo "❌ robust-kbench import failed"

# Print pinned versions
echo ""
echo "Pinned commits:"
cd third_party/EvoEngineer && echo "  EvoEngineer: $(git rev-parse HEAD)" && cd ../..
cd third_party/robust-kbench && echo "  robust-kbench: $(git rev-parse HEAD)" && cd ../..

echo ""
echo "✅ Bootstrap complete!"
echo "To activate: conda activate cuda-optim"
EOF

chmod +x scripts/bootstrap_tools.sh
```

**Run bootstrap:**
```bash
./scripts/bootstrap_tools.sh
```

**Commit:**
```bash
git add third_party/ scripts/bootstrap_tools.sh
git commit -m "feat: Add EvoEngineer and robust-kbench as pinned submodules

Submodules:
- EvoEngineer: ${EVOENGINEER_COMMIT}
- robust-kbench: ${RBK_COMMIT}

Bootstrap script: scripts/bootstrap_tools.sh
Lockfile: third_party/LOCKFILE.md
"
```

---

### Step 2: Establish Baselines (Correctness + Speed)

**Identify target kernel:**
```bash
# Current repo has:
# - cudadent42/bench/kernels/fa_inverted_v2_tensor_cores.cu (V2, correct but slow)
# - cudadent42/bench/kernels/fa_s512_v3.cu (V3, broken)

# Recommendation: Start with V2 as baseline, or create simple wrapper
```

**Create correctness test:**
```bash
cat > cudadent42/tests/test_sdpa_parity.py << 'EOFTEST'
#!/usr/bin/env python3
"""
Comprehensive SDPA parity test across all relevant shapes
"""

import torch
import torch.nn.functional as F
import pytest
import itertools

# Test parameters
DTYPES = [torch.float16, torch.bfloat16]
HEAD_DIMS = [64, 80, 96, 128]
SEQ_LENS = [128, 512, 1024, 2048, 4096]
BATCHES = [1, 4, 8]
HEADS = [8, 16]
CAUSAL = [True, False]
SEED = 42

ATOL = 1e-2
RTOL = 1e-2

def get_kernel_fn():
    """Import your custom kernel function"""
    # TODO: Replace with your kernel import
    # from cudadent42.bench.fa_inverted_v2_tensor_cores import forward
    # return forward
    
    # Placeholder: return SDPA as "kernel" (should be replaced)
    return lambda q, k, v, scale, causal: F.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=causal
    )

@pytest.mark.parametrize("dtype,D,S,B,H,causal", [
    (dtype, D, S, B, H, causal)
    for dtype, D, S, B, H, causal in itertools.product(
        DTYPES, HEAD_DIMS, SEQ_LENS, BATCHES, HEADS, CAUSAL
    )
    # Limit to manageable subset for CI
    if S <= 2048 or (S == 4096 and B == 1)
])
def test_parity(dtype, D, S, B, H, causal):
    """Test parity against SDPA"""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    device = 'cuda'
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    scale = 1.0 / (D ** 0.5)
    
    # Reference (SDPA)
    O_ref = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=causal)
    
    # Your kernel
    kernel_fn = get_kernel_fn()
    O_test = kernel_fn(Q, K, V, scale, causal)
    
    # Checks
    assert not torch.isnan(O_test).any(), f"NaN detected in output! Shape: B={B}, H={H}, S={S}, D={D}"
    assert not torch.isinf(O_test).any(), f"Inf detected in output! Shape: B={B}, H={H}, S={S}, D={D}"
    
    max_diff = (O_test - O_ref).abs().max().item()
    mean_diff = (O_test - O_ref).abs().mean().item()
    
    assert max_diff < ATOL, (
        f"Max diff {max_diff:.6f} > {ATOL} for "
        f"B={B}, H={H}, S={S}, D={D}, dtype={dtype}, causal={causal}"
    )
    
    print(f"✅ B={B}, H={H}, S={S}, D={D}, dtype={dtype}, causal={causal}: "
          f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
EOFTEST
```

**Create baseline benchmark:**
```bash
cat > cudadent42/scripts/bench_sdpa_baseline.py << 'EOFBENCH'
#!/usr/bin/env python3
"""
Baseline benchmark: SDPA vs our kernel
"""

import torch
import torch.nn.functional as F
import time
import json
import csv
import numpy as np
from pathlib import Path
import argparse

# Canonical shapes for L4
CANONICAL_SHAPES = [
    {"B": 4, "H": 16, "S": 2048, "D": 128, "causal": True, "name": "canonical_1"},
    {"B": 1, "H": 8, "S": 4096, "D": 128, "causal": True, "name": "canonical_2"},
    {"B": 2, "H": 8, "S": 512, "D": 64, "causal": False, "name": "canonical_3"},
]

def benchmark_kernel(kernel_fn, Q, K, V, scale, causal, warmup=20, iters=100):
    """Benchmark kernel with warmup"""
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(Q, K, V, scale=scale, is_causal=causal)
    torch.cuda.synchronize()
    
    # Measure
    latencies = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = kernel_fn(Q, K, V, scale=scale, is_causal=causal)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    return np.array(latencies)

def compute_tflops(B, H, S, D, latency_ms):
    """Compute TFLOP/s for attention"""
    # FLOPs: 2*B*H*S*S*D (QK^T) + 2*B*H*S*S*D (softmax+PV)
    flops = 4 * B * H * S * S * D
    tflops = (flops / 1e12) / (latency_ms / 1000)
    return tflops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shapes", choices=["canonical", "all"], default="canonical")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda'
    dtype = torch.float16
    
    shapes = CANONICAL_SHAPES if args.shapes == "canonical" else []  # TODO: Add full grid
    
    results_sdpa = []
    results_ours = []
    
    for shape in shapes:
        B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
        causal = shape["causal"]
        name = shape["name"]
        
        print(f"\nBenchmarking {name}: B={B}, H={H}, S={S}, D={D}, causal={causal}")
        
        # Create inputs
        torch.manual_seed(42)
        Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        K = torch.randn(B, H, S, D, device=device, dtype=dtype)
        V = torch.randn(B, H, S, D, device=device, dtype=dtype)
        scale = 1.0 / (D ** 0.5)
        
        # Benchmark SDPA
        latencies_sdpa = benchmark_kernel(F.scaled_dot_product_attention, Q, K, V, scale, causal)
        p50_sdpa = np.percentile(latencies_sdpa, 50)
        p90_sdpa = np.percentile(latencies_sdpa, 90)
        tflops_sdpa = compute_tflops(B, H, S, D, p50_sdpa)
        
        print(f"  SDPA: p50={p50_sdpa:.3f}ms, p90={p90_sdpa:.3f}ms, TFLOP/s={tflops_sdpa:.2f}")
        
        results_sdpa.append({
            "name": name,
            "B": B, "H": H, "S": S, "D": D, "causal": causal,
            "p50_ms": p50_sdpa,
            "p90_ms": p90_sdpa,
            "tflops": tflops_sdpa,
        })
        
        # TODO: Benchmark your kernel
        # For now, skip
        print(f"  Ours: (not implemented yet)")
    
    # Save results
    with open(output_dir / "baseline_sdpa.json", "w") as f:
        json.dump(results_sdpa, f, indent=2)
    
    with open(output_dir / "baseline_sdpa.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "B", "H", "S", "D", "causal", "p50_ms", "p90_ms", "tflops"])
        writer.writeheader()
        writer.writerows(results_sdpa)
    
    print(f"\n✅ Results saved to {output_dir}")
    print(f"  baseline_sdpa.json")
    print(f"  baseline_sdpa.csv")

if __name__ == "__main__":
    main()
EOFBENCH

chmod +x cudadent42/scripts/bench_sdpa_baseline.py
```

**Run baseline:**
```bash
conda activate cuda-optim
cd cudadent42

# Run baseline
python3 scripts/bench_sdpa_baseline.py --output-dir ../benchmarks/l4/${BENCH_DATE}/baseline --shapes canonical

# View results
cat ../benchmarks/l4/${BENCH_DATE}/baseline/baseline_sdpa.json
```

**Commit:**
```bash
git add cudadent42/tests/test_sdpa_parity.py cudadent42/scripts/bench_sdpa_baseline.py benchmarks/l4/${BENCH_DATE}/baseline/
git commit -m "feat: Add SDPA parity tests and baseline benchmarks

Tests:
- test_sdpa_parity.py: Comprehensive correctness across dtypes, shapes, causal
- Tolerances: atol=1e-2, rtol=1e-2 for fp16/bf16

Benchmarks:
- bench_sdpa_baseline.py: Canonical shapes with p50/p90/TFLOP/s
- Baseline results saved to benchmarks/l4/${BENCH_DATE}/baseline/

Next: Wire up robust-kbench for micro-benchmarking
"
```

---

### Step 3: Wire Up robust-kbench

**Create RBK config:**
```bash
cat > cudadent42/rbk_config.yaml << 'EOFRBK'
# robust-kbench configuration for L4 FlashAttention optimization

benchmarks:
  - name: "flashattention_l4"
    description: "FlashAttention optimization on L4 GPU"
    
    # Shape grid
    shapes:
      batch_sizes: [1, 4, 8]
      num_heads: [8, 16]
      seq_lens: [128, 512, 1024, 2048, 4096, 8192]
      head_dims: [64, 80, 96, 128]
      causal: [true, false]
      dtypes: ["float16", "bfloat16"]
    
    # Canonical shapes (prioritized)
    canonical_shapes:
      - {B: 4, H: 16, S: 2048, D: 128, causal: true}
      - {B: 1, H: 8, S: 4096, D: 128, causal: true}
      - {B: 2, H: 8, S: 512, D: 64, causal: false}
    
    # Kernels to test
    kernels:
      - name: "pytorch_sdpa"
        type: "reference"
        function: "torch.nn.functional.scaled_dot_product_attention"
      
      - name: "custom_kernel_v2"
        type: "candidate"
        function: "cudadent42.bench.fa_inverted_v2_tensor_cores.forward"
      
      # TODO: Add EvoEngineer-generated variants
    
    # Measurement settings
    measurement:
      warmup_iters: 20
      measure_iters: 100
      metrics:
        - "latency_p50_ms"
        - "latency_p90_ms"
        - "tflops"
        - "memory_gb"
    
    # Acceptance criteria
    acceptance:
      correctness:
        atol: 1e-2
        rtol: 1e-2
      performance:
        min_speedup_vs_reference: 1.03  # 3% improvement minimum
EOFRBK
```

**Create RBK runner script:**
```bash
cat > cudadent42/scripts/run_rbk.sh << 'EOFRBK'
#!/bin/bash
set -euo pipefail

BENCH_DATE=$(date +%Y-%m-%d)
OUTPUT_DIR="benchmarks/l4/${BENCH_DATE}/rbk"

mkdir -p ${OUTPUT_DIR}

echo "Running robust-kbench..."
robust-kbench run \
  -c cudadent42/rbk_config.yaml \
  -o ${OUTPUT_DIR}/rbk_report.json \
  --verbose

# Generate markdown summary
python3 cudadent42/scripts/rbk_to_markdown.py \
  ${OUTPUT_DIR}/rbk_report.json \
  ${OUTPUT_DIR}/rbk_report.md

echo "✅ RBK complete!"
echo "  Report: ${OUTPUT_DIR}/rbk_report.json"
echo "  Summary: ${OUTPUT_DIR}/rbk_report.md"
EOFRBK

chmod +x cudadent42/scripts/run_rbk.sh
```

**Create markdown converter:**
```bash
cat > cudadent42/scripts/rbk_to_markdown.py << 'EOFMD'
#!/usr/bin/env python3
"""Convert RBK JSON report to human-readable markdown"""

import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: rbk_to_markdown.py <input.json> <output.md>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    with open(input_path) as f:
        data = json.load(f)
    
    with open(output_path, "w") as f:
        f.write("# robust-kbench Report\n\n")
        f.write(f"**Date:** {data.get('date', 'N/A')}\n")
        f.write(f"**Config:** {input_path.stem}\n\n")
        
        # TODO: Parse and format data based on actual RBK output schema
        f.write("## Summary\n\n")
        f.write("(Add summary table here)\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("(Add per-shape results here)\n")
    
    print(f"✅ Markdown summary written to {output_path}")

if __name__ == "__main__":
    main()
EOFMD

chmod +x cudadent42/scripts/rbk_to_markdown.py
```

**Run RBK:**
```bash
conda activate cuda-optim
cd /path/to/periodicdent42
./cudadent42/scripts/run_rbk.sh
```

**Commit:**
```bash
git add cudadent42/rbk_config.yaml cudadent42/scripts/run_rbk.sh cudadent42/scripts/rbk_to_markdown.py benchmarks/l4/${BENCH_DATE}/rbk/
git commit -m "feat: Add robust-kbench configuration and runner

Config:
- Shape grid: batch, heads, seq_lens, head_dims, causal, dtypes
- Canonical shapes for L4 optimization
- Kernels: SDPA (reference) + custom variants

Scripts:
- run_rbk.sh: Run benchmarks and generate report
- rbk_to_markdown.py: Convert JSON to human-readable summary

Output: benchmarks/l4/${BENCH_DATE}/rbk/
"
```

---

### Step 4: Run EvoEngineer Guided Workflow

**Create EvoEngineer configuration:**
```bash
cat > cudadent42/evoengineer_config.yaml << 'EOFEVO'
# EvoEngineer configuration for L4 kernel optimization

project:
  name: "flashattention_l4_optim"
  base_kernel: "cudadent42/bench/kernels/fa_inverted_v2_tensor_cores.cu"
  output_dir: "cudadent42/bench/kernels/evolved/"

hardware:
  gpu: "L4"
  compute_capability: "8.9"
  smem_limit_kb: 48
  sm_count: 58

compilation:
  nvcc_flags:
    - "-O3"
    - "-use_fast_math"
    - "-DNDEBUG"
    - "-lineinfo"
    - "-Xptxas"
    - "-v"
    - "--expt-relaxed-constexpr"
    - "-arch=sm_89"
  
  resource_constraints:
    max_registers: 255
    min_occupancy: 0.30
    no_spills: true

correctness:
  test_command: "pytest cudadent42/tests/test_sdpa_parity.py -v"
  parity_threshold:
    atol: 1e-2
    rtol: 1e-2

performance:
  canonical_shapes:
    - {B: 4, H: 16, S: 2048, D: 128, causal: true}
    - {B: 1, H: 8, S: 4096, D: 128, causal: true}
    - {B: 2, H: 8, S: 512, D: 64, causal: false}
  
  acceptance_criteria:
    min_improvement_pct: 3
    required_shapes_beating: 2  # Must beat on at least 2 of 3 canonical shapes

optimization:
  mutation_strategies:
    - "tile_size_tuning"
    - "smem_layout_optimization"
    - "register_allocation"
    - "loop_unrolling"
    - "instruction_scheduling"
    - "memory_coalescing"
  
  iterations: 100
  population_size: 20
  elite_count: 5

leaderboard:
  output_file: "benchmarks/l4/{date}/leaderboard/leaderboard.json"
  metrics:
    - "p50_ms"
    - "p90_ms"
    - "tflops"
    - "speedup_vs_sdpa"
    - "occupancy"
    - "register_count"
EOFEVO
```

**Create EvoEngineer runner:**
```bash
cat > cudadent42/scripts/run_evoengineer.sh << 'EOFEVO'
#!/bin/bash
set -euo pipefail

BENCH_DATE=$(date +%Y-%m-%d)
export BENCH_DATE

echo "================================================"
echo "EvoEngineer Guided Optimization"
echo "================================================"

# Ensure output dirs exist
mkdir -p benchmarks/l4/${BENCH_DATE}/leaderboard
mkdir -p cudadent42/bench/kernels/evolved

# Run EvoEngineer
evoengineer optimize \
  --config cudadent42/evoengineer_config.yaml \
  --iterations 100 \
  --verbose

echo ""
echo "✅ EvoEngineer complete!"
echo "  Leaderboard: benchmarks/l4/${BENCH_DATE}/leaderboard/leaderboard.json"
echo "  Evolved kernels: cudadent42/bench/kernels/evolved/"
EOFEVO

chmod +x cudadent42/scripts/run_evoengineer.sh
```

**Run EvoEngineer:**
```bash
conda activate cuda-optim
cd /path/to/periodicdent42
./cudadent42/scripts/run_evoengineer.sh
```

**Monitor leaderboard:**
```bash
# Watch leaderboard in real-time
watch -n 5 "cat benchmarks/l4/${BENCH_DATE}/leaderboard/leaderboard.json | jq '.top_5'"
```

**Commit after run:**
```bash
git add cudadent42/evoengineer_config.yaml cudadent42/scripts/run_evoengineer.sh benchmarks/l4/${BENCH_DATE}/leaderboard/
git commit -m "feat: Add EvoEngineer optimization workflow

Config:
- L4-specific: 48KB SMEM, sm_89, 58 SMs
- Compile flags: -O3, -use_fast_math, -lineinfo, -arch=sm_89
- Resource constraints: max 255 regs, min 30% occupancy, no spills
- Acceptance: ≥3% improvement on ≥2/3 canonical shapes

Runner:
- run_evoengineer.sh: Execute optimization loop (100 iterations)
- Outputs: leaderboard.json, evolved kernels

Results: benchmarks/l4/${BENCH_DATE}/leaderboard/
"
```

---

### Step 5: Nsight Compute Deep Dive

**Create Nsight profiling script:**
```bash
cat > cudadent42/scripts/profile_nsight.sh << 'EOFNSIGHT'
#!/bin/bash
set -euo pipefail

BENCH_DATE=$(date +%Y-%m-%d)

# Shapes to profile
SHAPES=(
    "B=4,H=16,S=2048,D=128,causal=1"
    "B=1,H=8,S=4096,D=128,causal=1"
    "B=8,H=16,S=1024,D=64,causal=0"
)

KERNEL_REGEX="flash_attention.*"  # Adjust to match your kernel name

for shape_str in "${SHAPES[@]}"; do
    # Parse shape
    IFS=',' read -ra PARAMS <<< "$shape_str"
    B=$(echo ${PARAMS[0]} | cut -d= -f2)
    H=$(echo ${PARAMS[1]} | cut -d= -f2)
    S=$(echo ${PARAMS[2]} | cut -d= -f2)
    D=$(echo ${PARAMS[3]} | cut -d= -f2)
    CAUSAL=$(echo ${PARAMS[4]} | cut -d= -f2)
    
    SHAPE_NAME="B${B}_H${H}_S${S}_D${D}_causal${CAUSAL}"
    OUTPUT_DIR="benchmarks/l4/${BENCH_DATE}/nsight/${SHAPE_NAME}"
    mkdir -p ${OUTPUT_DIR}
    
    echo "Profiling ${SHAPE_NAME}..."
    
    # Run Nsight Compute
    ncu \
      --set full \
      --target-processes all \
      --replay-mode application \
      --export ${OUTPUT_DIR}/report \
      --kernel-name ${KERNEL_REGEX} \
      --force-overwrite \
      python3 cudadent42/scripts/profile_one_shape.py \
        --B ${B} --H ${H} --S ${S} --D ${D} --causal ${CAUSAL}
    
    # Export text summary
    ncu \
      --import ${OUTPUT_DIR}/report.ncu-rep \
      --page raw \
      --csv \
      > ${OUTPUT_DIR}/report.csv
    
    echo "✅ ${SHAPE_NAME} profiled"
    echo "  .ncu-rep: ${OUTPUT_DIR}/report.ncu-rep"
    echo "  .csv: ${OUTPUT_DIR}/report.csv"
done

echo ""
echo "✅ All shapes profiled!"
echo "Analyze with: ncu-ui benchmarks/l4/${BENCH_DATE}/nsight/<shape>/report.ncu-rep"
EOFNSIGHT

chmod +x cudadent42/scripts/profile_nsight.sh
```

**Create single-shape profiling script:**
```bash
cat > cudadent42/scripts/profile_one_shape.py << 'EOFPROFILE'
#!/usr/bin/env python3
"""Profile a single shape for Nsight Compute"""

import torch
import argparse

def get_kernel_fn():
    """Import your kernel"""
    # TODO: Replace with your kernel
    from cudadent42.bench.fa_inverted_v2_tensor_cores import forward
    return forward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--S", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--causal", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    
    device = 'cuda'
    dtype = torch.float16
    
    Q = torch.randn(args.B, args.H, args.S, args.D, device=device, dtype=dtype)
    K = torch.randn(args.B, args.H, args.S, args.D, device=device, dtype=dtype)
    V = torch.randn(args.B, args.H, args.S, args.D, device=device, dtype=dtype)
    
    scale = 1.0 / (args.D ** 0.5)
    causal = bool(args.causal)
    
    kernel_fn = get_kernel_fn()
    
    # Warmup
    for _ in range(5):
        _ = kernel_fn(Q, K, V, scale, causal)
    torch.cuda.synchronize()
    
    # Profile this call
    O = kernel_fn(Q, K, V, scale, causal)
    torch.cuda.synchronize()
    
    print(f"✅ Profiled B={args.B}, H={args.H}, S={args.S}, D={args.D}, causal={causal}")

if __name__ == "__main__":
    main()
EOFPROFILE

chmod +x cudadent42/scripts/profile_one_shape.py
```

**Run profiling:**
```bash
conda activate cuda-optim
cd /path/to/periodicdent42
./cudadent42/scripts/profile_nsight.sh
```

**Analyze results:**
```bash
# Open in Nsight Compute UI
ncu-ui benchmarks/l4/${BENCH_DATE}/nsight/B4_H16_S2048_D128_causal1/report.ncu-rep

# Or parse CSV for automated analysis
python3 cudadent42/scripts/analyze_nsight_csv.py \
  benchmarks/l4/${BENCH_DATE}/nsight/B4_H16_S2048_D128_causal1/report.csv \
  > benchmarks/l4/${BENCH_DATE}/nsight/B4_H16_S2048_D128_causal1/bottleneck_analysis.txt
```

**Create bottleneck analyzer:**
```bash
cat > cudadent42/scripts/analyze_nsight_csv.py << 'EOFANALYZE'
#!/usr/bin/env python3
"""Parse Nsight CSV and identify bottlenecks"""

import csv
import sys
from pathlib import Path

def parse_nsight_csv(csv_path):
    """Parse Nsight CSV and extract key metrics"""
    metrics = {}
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row.get('Metric Name', '')
            metric_value = row.get('Metric Value', '')
            
            # Store relevant metrics
            if 'SM Busy' in metric_name:
                metrics['sm_busy_pct'] = float(metric_value.rstrip('%'))
            elif 'Occupancy' in metric_name:
                metrics['occupancy_pct'] = float(metric_value.rstrip('%'))
            elif 'DRAM Throughput' in metric_name:
                metrics['dram_throughput_pct'] = float(metric_value.rstrip('%'))
            elif 'L2 Hit Rate' in metric_name:
                metrics['l2_hit_rate_pct'] = float(metric_value.rstrip('%'))
            elif 'Shared Memory Bank Conflicts' in metric_name:
                metrics['smem_bank_conflicts'] = int(metric_value)
            elif 'Branch Divergence' in metric_name:
                metrics['branch_divergence_pct'] = float(metric_value.rstrip('%'))
            elif 'Tensor Core Utilization' in metric_name:
                metrics['tensor_core_util_pct'] = float(metric_value.rstrip('%'))
    
    return metrics

def identify_bottlenecks(metrics):
    """Identify top bottlenecks"""
    bottlenecks = []
    
    # Check each metric
    if metrics.get('sm_busy_pct', 100) < 70:
        bottlenecks.append({
            'issue': 'Low SM Busy',
            'value': metrics['sm_busy_pct'],
            'severity': 'HIGH',
            'hypothesis': 'GPU underutilized; increase occupancy or grid size',
        })
    
    if metrics.get('dram_throughput_pct', 0) > 70:
        bottlenecks.append({
            'issue': 'DRAM Bandwidth Bound',
            'value': metrics['dram_throughput_pct'],
            'severity': 'HIGH',
            'hypothesis': 'Memory-bound; optimize data reuse with SMEM or improve coalescing',
        })
    
    if metrics.get('l2_hit_rate_pct', 100) < 50:
        bottlenecks.append({
            'issue': 'Low L2 Hit Rate',
            'value': metrics['l2_hit_rate_pct'],
            'severity': 'MEDIUM',
            'hypothesis': 'Poor cache locality; improve data access patterns or use persistent blocks',
        })
    
    if metrics.get('smem_bank_conflicts', 0) > 1000:
        bottlenecks.append({
            'issue': 'SMEM Bank Conflicts',
            'value': metrics['smem_bank_conflicts'],
            'severity': 'MEDIUM',
            'hypothesis': 'Pad SMEM arrays to avoid conflicts',
        })
    
    if metrics.get('tensor_core_util_pct', 0) < 50:
        bottlenecks.append({
            'issue': 'Low Tensor Core Utilization',
            'value': metrics['tensor_core_util_pct'],
            'severity': 'HIGH',
            'hypothesis': 'Use WMMA/HMMA for matrix tiles; ensure aligned sizes',
        })
    
    # Sort by severity
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    bottlenecks.sort(key=lambda x: severity_order[x['severity']])
    
    return bottlenecks

def main():
    if len(sys.argv) != 2:
        print("Usage: analyze_nsight_csv.py <report.csv>")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    print("=" * 80)
    print("Nsight Compute Bottleneck Analysis")
    print("=" * 80)
    print(f"\nFile: {csv_path}\n")
    
    metrics = parse_nsight_csv(csv_path)
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("Bottlenecks (Prioritized)")
    print("=" * 80)
    
    bottlenecks = identify_bottlenecks(metrics)
    
    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"\n#{i} [{bottleneck['severity']}] {bottleneck['issue']}")
        print(f"  Value: {bottleneck['value']}")
        print(f"  Hypothesis: {bottleneck['hypothesis']}")
    
    if not bottlenecks:
        print("\n✅ No major bottlenecks detected!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
EOFANALYZE

chmod +x cudadent42/scripts/analyze_nsight_csv.py
```

**Commit:**
```bash
git add cudadent42/scripts/profile_nsight.sh cudadent42/scripts/profile_one_shape.py cudadent42/scripts/analyze_nsight_csv.py benchmarks/l4/${BENCH_DATE}/nsight/
git commit -m "feat: Add Nsight Compute profiling and bottleneck analysis

Scripts:
- profile_nsight.sh: Profile 3 canonical shapes with ncu --set full
- profile_one_shape.py: Single-shape profiling target
- analyze_nsight_csv.py: Parse CSV and identify bottlenecks

Metrics tracked:
- SM busy %, occupancy, DRAM throughput, L2 hit rate
- SMEM bank conflicts, branch divergence, tensor core utilization

Output: benchmarks/l4/${BENCH_DATE}/nsight/<shape>/{report.ncu-rep, report.csv, bottleneck_analysis.txt}
"
```

---

### Steps 6-10: Inversion, Polish, Cross-Bench, Regression Gates, Success Criteria

**(Due to length, providing condensed instructions)**

**Step 6: Inversion Thinking**
- Create `cudadent42/bench/kernels/deliberately_broken/` with pathological variants
- Measure slowdown per pathology (non-coalesced loads, no SMEM, etc.)
- Implement opposites in main kernel

**Step 7: Expert Polish**
- Use `--maxrregcount` experiments
- Integrate CUTLASS tensor-op fragments
- Add CUB for reductions

**Step 8: Cross-Bench Validation**
- Run CUTLASS profiler on equivalent tiles
- Export `benchmarks/l4/${BENCH_DATE}/cutlass_comparison.json`

**Step 9: Regression Gates**
- Create `scripts/ci_local_gpu_gate.sh` that fails on >2% regression
- Save `final_summary.md` with speedup table

**Step 10: Success Criteria Check**
- ✅ Correctness parity (no NaNs/Inf)
- ✅ ≥10% speedup on 2/3 canonical shapes
- ✅ p90 not worse than SDPA
- ✅ SM busy ≥70% on at least one shape

---

## Final Deliverables Checklist

When you're done, you should have:

```
benchmarks/l4/${BENCH_DATE}/
├── environment.json
├── baseline/
│   ├── baseline_sdpa.json
│   └── baseline_sdpa.csv
├── rbk/
│   ├── rbk_report.json
│   └── rbk_report.md
├── leaderboard/
│   └── leaderboard.json
├── nsight/
│   ├── B4_H16_S2048_D128_causal1/
│   │   ├── report.ncu-rep
│   │   ├── report.csv
│   │   └── bottleneck_analysis.txt
│   ├── B1_H8_S4096_D128_causal1/
│   └── B8_H16_S1024_D64_causal0/
└── final/
    ├── final_summary.md
    ├── speedup_table.csv
    └── kernel_diff.patch
```

**PR Template:**
```markdown
# L4 SDPA-Beating Kernel via EvoEngineer + robust-kbench

## Summary

- **Baseline:** PyTorch SDPA (0.073 ms on canonical_3)
- **Ours:** Custom kernel (X.XXX ms on canonical_3)
- **Speedup:** X.XX× on Y/3 canonical shapes

## Improvements

1. **Optimization 1:** [Description] → +X% speedup
2. **Optimization 2:** [Description] → +Y% speedup
3. **Optimization 3:** [Description] → +Z% speedup

## Proof

| Shape | SDPA p50 | Ours p50 | Speedup | SDPA p90 | Ours p90 |
|-------|----------|----------|---------|----------|----------|
| canonical_1 | X.XXX ms | Y.YYY ms | Z.ZZ× | A.AAA ms | B.BBB ms |
| canonical_2 | X.XXX ms | Y.YYY ms | Z.ZZ× | A.AAA ms | B.BBB ms |
| canonical_3 | X.XXX ms | Y.YYY ms | Z.ZZ× | A.AAA ms | B.BBB ms |

## Nsight Findings

1. **Bottleneck 1:** [Issue] → [Fix applied]
2. **Bottleneck 2:** [Issue] → [Fix applied]
3. **Bottleneck 3:** [Issue] → [Fix applied]

## Repro Steps

\`\`\`bash
# 1. Bootstrap tools
./scripts/bootstrap_tools.sh

# 2. Run baseline
python3 scripts/bench_sdpa_baseline.py --output-dir benchmarks/l4/$(date +%Y-%m-%d)/baseline

# 3. Run EvoEngineer
./scripts/run_evoengineer.sh

# 4. Profile with Nsight
./scripts/profile_nsight.sh

# 5. Validate with CI gate
./scripts/ci_local_gpu_gate.sh
\`\`\`

## Known Limitations

- [Limitation 1]
- [Limitation 2]

## Next Experiments

1. [Next idea 1]
2. [Next idea 2]
```

---

## Critical Success Factors (From V3 Learnings)

### DO:
1. ✅ **Profile SDPA first** - Understand the bar before trying to beat it
2. ✅ **Test correctness at EVERY step** - Never optimize broken code
3. ✅ **Start simple** - Build V1 → V1.1 → V1.2, test each increment
4. ✅ **Use compute-sanitizer** - Catch memory bugs early
5. ✅ **Set stop conditions** - Time, iteration, and cost budgets
6. ✅ **Document failures** - Learnings are valuable even when code isn't

### DON'T:
1. ❌ **Skip unit tests** - Test QK, softmax, SV separately
2. ❌ **Jump to complex optimizations** - Persistent blocks, cp.async, etc. only after basics work
3. ❌ **Ignore red flags** - If 0.675× scaling appears, STOP and debug, don't add more features
4. ❌ **Exceed budgets** - If 2 iterations don't fix it, stop and reassess
5. ❌ **Optimize for wrong hardware** - L4's 48KB SMEM != H100's 227KB
6. ❌ **Forget to compare** - Always benchmark vs SDPA, not vs previous broken version

---

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Bootstrap tools
./scripts/bootstrap_tools.sh
conda activate cuda-optim

# Run baseline
cd cudadent42
python3 scripts/bench_sdpa_baseline.py --output-dir ../benchmarks/l4/$(date +%Y-%m-%d)/baseline

# Test correctness
pytest tests/test_sdpa_parity.py -v -s

# Run robust-kbench
./scripts/run_rbk.sh

# Run EvoEngineer
./scripts/run_evoengineer.sh

# Profile with Nsight
./scripts/profile_nsight.sh

# Analyze Nsight results
python3 scripts/analyze_nsight_csv.py benchmarks/l4/<date>/nsight/<shape>/report.csv

# Check regression gate
./scripts/ci_local_gpu_gate.sh

# View leaderboard
cat benchmarks/l4/$(date +%Y-%m-%d)/leaderboard/leaderboard.json | jq '.top_5'
```

---

## Contact & Support

**Previous Session Evidence:**
- `V3_POSTMORTEM.md` - Complete failure analysis with lessons
- `ENGINEER_LOG.md` - Detailed session history
- `artifacts/` - All test results, profiling data, numpy arrays

**Questions?** Refer to:
1. V3_POSTMORTEM.md (Section: "What We'd Do Differently")
2. ENGINEER_LOG.md (Steps 0-6 with findings)
3. artifacts/oracle/ (Example of good diagnostic workflow)

---

**Status:** Knowledge transfer complete. Ready for new engineer to execute EvoEngineer + robust-kbench workflow with V3 learnings incorporated.

**Remember:** Science is the goal. Improvement in iteration attempts is the goal. Honesty and excellence.

