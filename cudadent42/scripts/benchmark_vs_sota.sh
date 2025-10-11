#!/bin/bash
# Systematic SOTA Benchmark Execution Script
# Compares CUDAdent42 FlashAttention vs PyTorch SDPA (industry baseline)

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 CUDAdent42 vs PyTorch SDPA: SOTA Benchmark Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Baseline: PyTorch 2.x F.scaled_dot_product_attention (flash_sdp backend)"
echo "Target: CUDAdent42 FlashAttention (Phase 2 implementation)"
echo "Hardware: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

# Activate environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
fi

# Install dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Installing dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install torch torchvision numpy pytest --quiet

# Build library using manual build (proven working)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Building CUDAdent42 library (manual build)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd cudadent42
bash build_manual.sh

# Run correctness tests first
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Running correctness validation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 tests/test_correctness.py || { echo "❌ Correctness tests failed. Aborting benchmark."; exit 1; }

# Run comprehensive SOTA comparison benchmarks
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Running SOTA Comparison Benchmarks..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration Matrix:"
echo "  • Sequence lengths: 128, 512, 1024, 2048, 4096"
echo "  • Head dimensions: 32, 64, 128"
echo "  • Batch sizes: 1, 4, 8"
echo "  • Data types: FP16, BF16 (if supported)"
echo "  • Repeats: 50 (warm-up: 10)"
echo ""

# Create output directory
mkdir -p benchmark_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmark_results/sota_comparison_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Run benchmark with full output
python3 benches/bench_correctness_and_speed.py \
    --output-dir "$OUTPUT_DIR" \
    --repeats 50 \
    --warmup 10 \
    --save-csv \
    --verbose \
    | tee "$OUTPUT_DIR/benchmark_log.txt"

# Generate summary report
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 Generating Summary Report..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat > "$OUTPUT_DIR/SUMMARY.md" << 'SUMMARY'
# CUDAdent42 vs PyTorch SDPA: SOTA Benchmark Results

## Methodology

**Baseline**: PyTorch 2.x `F.scaled_dot_product_attention` with `flash_sdp` backend
**Target**: CUDAdent42 FlashAttention Phase 2 implementation
**Hardware**: NVIDIA L4 GPU (SM89, 24GB VRAM)
**Date**: $(date)

**Statistical Rigor**:
- 50 repeated measurements per configuration
- 10 warm-up iterations (excluded from timing)
- CUDA events for precise timing
- Mean ± standard deviation reported
- Memory tracking via PyTorch CUDA allocator

## Test Matrix

| Parameter | Values |
|-----------|--------|
| Sequence lengths | 128, 512, 1024, 2048, 4096 |
| Head dimensions | 32, 64, 128 |
| Batch sizes | 1, 4, 8 |
| Data types | FP16, BF16 |
| Repeats | 50 (+ 10 warmup) |

## Results

### Performance Comparison

See `benchmark_results.csv` for raw data.

### Key Findings

1. **Correctness**: All tests pass numerical validation vs PyTorch SDPA (atol=1e-2, rtol=1e-3)
2. **Performance**: [TO BE FILLED FROM ACTUAL RESULTS]
3. **Memory**: [TO BE FILLED FROM ACTUAL RESULTS]
4. **Scalability**: [TO BE FILLED FROM ACTUAL RESULTS]

### Limitations

- Phase 2 implementation (baseline FlashAttention algorithm)
- No warp specialization yet (planned Phase 3)
- No async copy pipeline yet (planned Phase 3)
- Single-GPU only (no multi-GPU support)

### Expected vs Actual Performance

**Expected** (based on FlashAttention paper):
- Memory: O(N) vs O(N²) for standard attention
- Speed: 2-4x faster than PyTorch baseline for long sequences

**Actual**: [TO BE FILLED FROM BENCHMARK RUN]

## Reproducibility

To reproduce these results:

\`\`\`bash
cd cudadent42
bash build_manual.sh
python3 benches/bench_correctness_and_speed.py --repeats 50 --warmup 10
\`\`\`

## References

- PyTorch SDPA: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- FlashAttention: Dao et al., 2022 (https://arxiv.org/abs/2205.14135)
- FlashAttention-2: Dao et al., 2023 (https://arxiv.org/abs/2307.08691)

SUMMARY

# Collect GPU info
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUTPUT_DIR/gpu_info.csv"

# Save environment info
pip freeze > "$OUTPUT_DIR/requirements_used.txt"
nvcc --version > "$OUTPUT_DIR/cuda_version.txt" 2>&1 || echo "nvcc not in PATH" > "$OUTPUT_DIR/cuda_version.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Benchmark Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  • benchmark_log.txt - Full output"
echo "  • benchmark_results.csv - Raw data"
echo "  • SUMMARY.md - Summary report"
echo "  • gpu_info.csv - Hardware info"
echo ""
echo "Next: Review results and update README with actual measured performance"
echo ""

