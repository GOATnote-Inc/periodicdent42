#!/bin/bash
# ============================================================================
# BlackwellSparseK Remote H100 Deployment Automation
# ============================================================================
# Deploys and validates BlackwellSparseK on RunPod H100 instance
# Expert CUDA engineer focused on safety, determinism, reproducibility
# ============================================================================

set -euo pipefail

# Configuration
RUNPOD_IP="${RUNPOD_IP:-154.57.34.90}"
RUNPOD_PORT="${RUNPOD_PORT:-23673}"
RUNPOD_USER="root"
LOCAL_PROJECT="/Users/kiteboard/periodicdent42/BlackwellSparseK"
REMOTE_PROJECT="/workspace/BlackwellSparseK"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_info() { echo -e "${CYAN}ℹ️  $1${NC}"; }
log_step() { echo -e "${MAGENTA}▶ $1${NC}"; }

print_banner() {
    echo ""
    echo "=========================================="
    echo "  BlackwellSparseK H100 Deployment"
    echo "=========================================="
    echo "  Target: ${RUNPOD_IP}:${RUNPOD_PORT}"
    echo "  Timestamp: ${TIMESTAMP}"
    echo "  Mode: Remote Execution"
    echo "=========================================="
    echo ""
}

# SSH connection test with retry
test_ssh_connection() {
    log_step "Testing SSH connection to RunPod H100..."
    
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if ssh -p ${RUNPOD_PORT} \
               -o ConnectTimeout=10 \
               -o StrictHostKeyChecking=no \
               -o TCPKeepAlive=yes \
               -o ServerAliveInterval=20 \
               ${RUNPOD_USER}@${RUNPOD_IP} \
               "echo 'SSH OK'" &> /dev/null; then
            log_success "SSH connection established (attempt $attempt/$max_attempts)"
            return 0
        fi
        
        log_warning "SSH connection failed (attempt $attempt/$max_attempts), retrying..."
        sleep 5
        ((attempt++))
    done
    
    log_error "SSH connection failed after $max_attempts attempts"
    return 1
}

# Verify H100 GPU on remote
verify_h100_gpu() {
    log_step "Verifying H100 GPU on remote..."
    
    local gpu_info=$(ssh -p ${RUNPOD_PORT} \
                         -o StrictHostKeyChecking=no \
                         ${RUNPOD_USER}@${RUNPOD_IP} \
                         "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader" 2>&1)
    
    if echo "$gpu_info" | grep -q "H100"; then
        log_success "H100 GPU detected: $gpu_info"
        return 0
    else
        log_warning "Expected H100, found: $gpu_info"
        echo ""
        read -p "Continue anyway? [y/N]: " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            return 1
        fi
    fi
    
    return 0
}

# Clean remote workspace
clean_remote_workspace() {
    log_step "Cleaning remote workspace..."
    
    ssh -p ${RUNPOD_PORT} \
        -o StrictHostKeyChecking=no \
        ${RUNPOD_USER}@${RUNPOD_IP} \
        "rm -rf ${REMOTE_PROJECT}" || true
    
    log_success "Remote workspace cleaned"
}

# Deploy BlackwellSparseK to remote
deploy_to_remote() {
    log_step "Deploying BlackwellSparseK to H100..."
    
    # Create deployment tarball
    log_info "Creating deployment package..."
    cd "$(dirname ${LOCAL_PROJECT})"
    
    local package_name="blackwell-sparsek-${TIMESTAMP}.tar.gz"
    tar -czf "${package_name}" \
        --exclude='BlackwellSparseK/.git' \
        --exclude='BlackwellSparseK/__pycache__' \
        --exclude='BlackwellSparseK/*.pyc' \
        --exclude='BlackwellSparseK/build' \
        --exclude='BlackwellSparseK/dist' \
        --exclude='BlackwellSparseK/*.egg-info' \
        --exclude='BlackwellSparseK/.vscode' \
        --exclude='BlackwellSparseK/validation.log' \
        BlackwellSparseK/
    
    log_success "Package created: ${package_name} ($(du -h ${package_name} | cut -f1))"
    
    # Upload to remote
    log_info "Uploading to RunPod H100..."
    scp -P ${RUNPOD_PORT} \
        -o StrictHostKeyChecking=no \
        -o TCPKeepAlive=yes \
        "${package_name}" \
        ${RUNPOD_USER}@${RUNPOD_IP}:/workspace/
    
    log_success "Upload complete"
    
    # Extract on remote
    log_info "Extracting on remote..."
    ssh -p ${RUNPOD_PORT} \
        -o StrictHostKeyChecking=no \
        ${RUNPOD_USER}@${RUNPOD_IP} \
        "cd /workspace && tar -xzf ${package_name} && rm ${package_name}"
    
    log_success "Deployment complete"
    
    # Cleanup local tarball
    rm -f "${package_name}"
}

# Execute H100 validation remotely
execute_remote_validation() {
    log_step "Executing H100 validation (7 loops)..."
    echo ""
    
    ssh -p ${RUNPOD_PORT} \
        -o StrictHostKeyChecking=no \
        -o TCPKeepAlive=yes \
        -o ServerAliveInterval=20 \
        ${RUNPOD_USER}@${RUNPOD_IP} \
        "cd ${REMOTE_PROJECT} && echo '1' | bash scripts/h100_orchestrator.sh"
    
    local exit_code=$?
    echo ""
    
    if [ $exit_code -eq 0 ]; then
        log_success "Remote validation completed successfully"
        return 0
    else
        log_error "Remote validation failed with exit code $exit_code"
        return $exit_code
    fi
}

# Download validation report
download_report() {
    log_step "Downloading validation report..."
    
    mkdir -p "${LOCAL_PROJECT}/results"
    
    scp -P ${RUNPOD_PORT} \
        -o StrictHostKeyChecking=no \
        ${RUNPOD_USER}@${RUNPOD_IP}:/workspace/results/H100_VALIDATION_REPORT.md \
        "${LOCAL_PROJECT}/results/H100_VALIDATION_REPORT_${TIMESTAMP}.md" 2>/dev/null || {
        log_warning "Could not download validation report"
        return 1
    }
    
    log_success "Report downloaded: results/H100_VALIDATION_REPORT_${TIMESTAMP}.md"
    
    # Also download logs
    scp -P ${RUNPOD_PORT} \
        -o StrictHostKeyChecking=no \
        ${RUNPOD_USER}@${RUNPOD_IP}:/workspace/results/validation.log \
        "${LOCAL_PROJECT}/results/validation_${TIMESTAMP}.log" 2>/dev/null || {
        log_warning "Could not download validation.log"
    }
    
    return 0
}

# Print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "  Deployment Summary"
    echo "=========================================="
    echo "  Target: ${RUNPOD_IP}:${RUNPOD_PORT}"
    echo "  Timestamp: ${TIMESTAMP}"
    echo ""
    
    if [ -f "${LOCAL_PROJECT}/results/H100_VALIDATION_REPORT_${TIMESTAMP}.md" ]; then
        log_success "Validation report available"
        echo ""
        echo "View report:"
        echo "  cat ${LOCAL_PROJECT}/results/H100_VALIDATION_REPORT_${TIMESTAMP}.md"
        echo ""
        echo "Last 20 lines:"
        tail -20 "${LOCAL_PROJECT}/results/H100_VALIDATION_REPORT_${TIMESTAMP}.md" | sed 's/^/  /'
    fi
    
    echo "=========================================="
}

# Main execution
main() {
    print_banner
    
    # Step 1: Test SSH connection
    if ! test_ssh_connection; then
        log_error "SSH connection failed. Aborting."
        echo ""
        log_info "Troubleshooting:"
        log_info "  1. Verify RunPod instance is running"
        log_info "  2. Check IP and port from RunPod dashboard"
        log_info "  3. Ensure SSH key is configured: ~/.ssh/id_rsa"
        exit 1
    fi
    
    echo ""
    
    # Step 2: Verify H100 GPU
    if ! verify_h100_gpu; then
        log_error "GPU verification failed. Aborting."
        exit 1
    fi
    
    echo ""
    
    # Step 3: Clean remote workspace
    clean_remote_workspace
    
    echo ""
    
    # Step 4: Deploy to remote
    if ! deploy_to_remote; then
        log_error "Deployment failed. Aborting."
        exit 1
    fi
    
    echo ""
    
    # Step 5: Execute validation
    if ! execute_remote_validation; then
        log_error "Validation failed."
        # Still try to download report for debugging
        download_report || true
        exit 1
    fi
    
    echo ""
    
    # Step 6: Download report
    download_report || true
    
    # Step 7: Print summary
    print_summary
    
    log_success "Deployment and validation complete!"
    echo ""
    log_info "Status: CLEARED FOR DEPLOYMENT ✅"
}

# Run main
main "$@"

