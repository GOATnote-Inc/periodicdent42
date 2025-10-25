#!/bin/bash
#
# Full Optimization Pipeline
# Executes all 4 phases of the integrated plan
#
# Estimated time: 2 hours
# Estimated cost: $1.36 (L4 GPU @ $0.68/hour)
#
# Author: Brandon Dent (b@thegoatnote.com)
# License: Apache 2.0

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_DIR="/home/bdent/periodicdent42"
ARTIFACTS_DIR="cudadent42/bench/artifacts"
BATCH=32
HEADS=8
SEQ=512
DIM=64
ITERATIONS=100
WARMUP=20

# Helper functions
log_phase() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    echo ""
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. GPU not available?"
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        log_error "nvidia-smi failed. GPU driver issue?"
        exit 1
    fi
    
    log_success "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
}

check_python_deps() {
    if ! python -c "import torch" &> /dev/null; then
        log_error "PyTorch not found. Run: pip install torch"
        exit 1
    fi
    
    if ! python -c "import numpy" &> /dev/null; then
        log_error "NumPy not found. Run: pip install numpy"
        exit 1
    fi
    
    if ! python -c "import scipy" &> /dev/null; then
        log_warning "SciPy not found. Some statistical features may be limited. Run: pip install scipy"
    fi
    
    log_success "Python dependencies OK"
}

# Main execution
main() {
    log_phase "FULL OPTIMIZATION PIPELINE"
    
    echo "Configuration:"
    echo "  Batch:      $BATCH"
    echo "  Heads:      $HEADS"
    echo "  Sequence:   $SEQ"
    echo "  Dimension:  $DIM"
    echo "  Iterations: $ITERATIONS"
    echo "  Warmup:     $WARMUP"
    echo ""
    echo "Estimated time: 2 hours"
    echo "Estimated cost: \$1.36 (L4 GPU)"
    echo ""
    
    # Checks
    log_phase "PHASE 0: ENVIRONMENT CHECKS"
    check_gpu
    check_python_deps
    echo ""
    
    # Navigate to repository
    cd "$REPO_DIR" || exit 1
    
    # Phase 1: Enhanced Benchmark
    log_phase "PHASE 1: ENHANCED BENCHMARK (15 min, \$0.17)"
    
    START_TIME=$(date +%s)
    
    python cudadent42/bench/integrated_test_enhanced.py \
        --batch $BATCH --heads $HEADS --seq $SEQ --dim $DIM \
        --iterations $ITERATIONS --warmup $WARMUP \
        --lock-env \
        --output-dir $ARTIFACTS_DIR
    
    PHASE1_TIME=$(($(date +%s) - START_TIME))
    log_success "Phase 1 complete in ${PHASE1_TIME}s"
    echo ""
    
    # Phase 2: Optimization Loop
    log_phase "PHASE 2: FIXED-SHAPE OPTIMIZATION (60 min, \$0.68)"
    
    START_TIME=$(date +%s)
    
    python cudadent42/bench/sota_optimization_loop.py \
        --batch $BATCH --heads $HEADS --seq $SEQ --dim $DIM \
        --budget-min 60 \
        --iterations $ITERATIONS \
        --warmup $WARMUP \
        --target-speedup 1.10 \
        --output-dir $ARTIFACTS_DIR/optimization
    
    PHASE2_TIME=$(($(date +%s) - START_TIME))
    log_success "Phase 2 complete in ${PHASE2_TIME}s"
    echo ""
    
    # Phase 3: Multi-Shape Comparison
    log_phase "PHASE 3: MULTI-SHAPE COMPARISON (30 min, \$0.34)"
    
    START_TIME=$(date +%s)
    
    python cudadent42/bench/integrated_test_enhanced.py \
        --batch $BATCH --heads $HEADS --dim $DIM \
        --seq 128 256 512 1024 \
        --iterations $ITERATIONS \
        --warmup $WARMUP \
        --compare \
        --output-dir $ARTIFACTS_DIR
    
    PHASE3_TIME=$(($(date +%s) - START_TIME))
    log_success "Phase 3 complete in ${PHASE3_TIME}s"
    echo ""
    
    # Phase 4: Generate Combined Report
    log_phase "PHASE 4: GENERATE COMBINED REPORT (15 min)"
    
    START_TIME=$(date +%s)
    
    python scripts/generate_combined_report.py \
        --artifacts-dir $ARTIFACTS_DIR \
        --output $ARTIFACTS_DIR/COMBINED_REPORT.md
    
    PHASE4_TIME=$(($(date +%s) - START_TIME))
    log_success "Phase 4 complete in ${PHASE4_TIME}s"
    echo ""
    
    # Summary
    log_phase "EXECUTION COMPLETE"
    
    TOTAL_TIME=$((PHASE1_TIME + PHASE2_TIME + PHASE3_TIME + PHASE4_TIME))
    TOTAL_HOURS=$(echo "scale=2; $TOTAL_TIME / 3600" | bc)
    TOTAL_COST=$(echo "scale=2; $TOTAL_HOURS * 0.68" | bc)
    
    echo "Summary:"
    echo "  Phase 1 (Enhanced Benchmark):    ${PHASE1_TIME}s"
    echo "  Phase 2 (Optimization Loop):     ${PHASE2_TIME}s"
    echo "  Phase 3 (Multi-Shape):            ${PHASE3_TIME}s"
    echo "  Phase 4 (Report Generation):     ${PHASE4_TIME}s"
    echo "  ─────────────────────────────────────────"
    echo "  Total Time:                       ${TOTAL_TIME}s ($TOTAL_HOURS hours)"
    echo "  Estimated Cost:                   \$$TOTAL_COST"
    echo ""
    
    log_success "All artifacts saved to: $ARTIFACTS_DIR"
    log_success "Combined report: $ARTIFACTS_DIR/COMBINED_REPORT.md"
    echo ""
    
    log_phase "NEXT STEPS"
    echo "1. Review combined report:  cat $ARTIFACTS_DIR/COMBINED_REPORT.md"
    echo "2. Copy to local machine:   gcloud compute scp cuda-dev:$REPO_DIR/$ARTIFACTS_DIR/ . --recurse --zone=us-central1-a"
    echo "3. (Optional) Run Nsight:   bash scripts/run_nsight_profiling.sh"
    echo "4. Stop GPU to save costs:  gcloud compute instances stop cuda-dev --zone=us-central1-a"
    echo ""
    
    log_success "Pipeline execution complete! 🎉"
}

# Execute
main "$@"

