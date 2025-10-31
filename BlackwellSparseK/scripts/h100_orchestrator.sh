#!/bin/bash
# ============================================================================
# BlackwellSparseK H100 Validation Orchestrator
# ============================================================================
# VS Code task-compatible wrapper for H100 validation
# Provides IDE-friendly output with progress indicators
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Output functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

log_step() {
    echo -e "${MAGENTA}▶ $1${NC}"
}

# Banner
print_banner() {
    echo ""
    echo "=========================================="
    echo "  BlackwellSparseK H100 Validation"
    echo "=========================================="
    echo "  Version: 0.1.0"
    echo "  Timestamp: ${TIMESTAMP}"
    echo "  Mode: VS Code Integrated"
    echo "=========================================="
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local errors=0
    
    # Check if in BlackwellSparseK directory
    if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
        log_error "Not in BlackwellSparseK directory"
        ((errors++))
    fi
    
    # Check for validation script
    if [ ! -f "${SCRIPT_DIR}/validate_h100_7loop.sh" ]; then
        log_error "validate_h100_7loop.sh not found"
        ((errors++))
    fi
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found (required for container build)"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "Prerequisites check passed"
        return 0
    else
        log_error "Prerequisites check failed ($errors errors)"
        return 1
    fi
}

# Execution mode selection
select_execution_mode() {
    # All prompts go to stderr to avoid polluting command substitution
    log_step "Select execution mode:" >&2
    echo "" >&2
    echo "  1) Local H100 (direct execution)" >&2
    echo "  2) Remote H100 (SSH to RunPod/Vast.ai)" >&2
    echo "  3) Dry run (check scripts only)" >&2
    echo "" >&2
    
    # Handle both interactive and non-interactive input
    if [ -t 0 ]; then
        # Interactive terminal
        read -p "Enter choice [1-3]: " choice
    else
        # Non-interactive (piped input)
        read choice
    fi
    echo "" >&2
    
    # Only the mode string goes to stdout for capture
    case $choice in
        1) echo "local" ;;
        2) echo "remote" ;;
        3) echo "dryrun" ;;
        *) echo "dryrun" ;;  # Default to dry run for safety
    esac
}

# Local execution
execute_local() {
    log_step "Executing local H100 validation..."
    
    # Check for GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Is CUDA installed?"
        return 1
    fi
    
    # Check for H100
    if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
        log_warning "H100 GPU not detected. Detected:"
        nvidia-smi --query-gpu=name --format=csv,noheader
        echo ""
        read -p "Continue anyway? [y/N]: " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            log_info "Validation cancelled by user"
            return 1
        fi
    fi
    
    # Execute validation
    log_info "Starting 7-loop validation framework..."
    echo ""
    
    cd "${PROJECT_ROOT}"
    bash "${SCRIPT_DIR}/validate_h100_7loop.sh"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Validation completed successfully"
        
        # Show report location
        if [ -f "/workspace/results/H100_VALIDATION_REPORT.md" ]; then
            echo ""
            log_info "Report available at: /workspace/results/H100_VALIDATION_REPORT.md"
        fi
    else
        log_error "Validation failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Remote execution
execute_remote() {
    log_step "Executing remote H100 validation..."
    
    # Get connection details
    echo ""
    read -p "Enter H100 IP address: " H100_IP
    read -p "Enter SSH port [default: 22]: " H100_PORT
    H100_PORT=${H100_PORT:-22}
    
    echo ""
    log_info "Connection: root@${H100_IP}:${H100_PORT}"
    
    # Test SSH connection
    log_step "Testing SSH connection..."
    if ssh -p ${H100_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        root@${H100_IP} "echo 'SSH OK'" &> /dev/null; then
        log_success "SSH connection successful"
    else
        log_error "SSH connection failed"
        log_info "Ensure SSH key is configured and port is correct"
        return 1
    fi
    
    # Create deployment package
    log_step "Creating deployment package..."
    cd "${PROJECT_ROOT}/.."
    
    local package_name="blackwell-sparsek-${TIMESTAMP}.tar.gz"
    tar -czf "${package_name}" \
        BlackwellSparseK/src \
        BlackwellSparseK/tests \
        BlackwellSparseK/benchmarks \
        BlackwellSparseK/examples \
        BlackwellSparseK/docker \
        BlackwellSparseK/scripts \
        BlackwellSparseK/pyproject.toml \
        BlackwellSparseK/setup.py \
        BlackwellSparseK/CMakeLists.txt \
        BlackwellSparseK/docker-compose.yml \
        BlackwellSparseK/README.md \
        BlackwellSparseK/LICENSE \
        BlackwellSparseK/CHANGELOG.md \
        BlackwellSparseK/docs
    
    log_success "Package created: ${package_name}"
    
    # Upload package
    log_step "Uploading to H100..."
    scp -P ${H100_PORT} -o StrictHostKeyChecking=no \
        "${package_name}" root@${H100_IP}:/workspace/
    log_success "Upload complete"
    
    # Execute remote validation
    log_step "Executing remote validation..."
    ssh -p ${H100_PORT} -o StrictHostKeyChecking=no root@${H100_IP} << 'ENDSSH'
        cd /workspace
        
        # Extract package
        tar -xzf blackwell-sparsek-*.tar.gz
        cd BlackwellSparseK
        
        # Run validation
        bash scripts/validate_h100_7loop.sh
        
        # Return exit code
        exit $?
ENDSSH
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Remote validation completed successfully"
        
        # Download results
        log_step "Downloading validation report..."
        mkdir -p "${PROJECT_ROOT}/results"
        scp -P ${H100_PORT} -o StrictHostKeyChecking=no \
            root@${H100_IP}:/workspace/results/H100_VALIDATION_REPORT.md \
            "${PROJECT_ROOT}/results/" 2>/dev/null || \
            log_warning "Could not download validation report"
    else
        log_error "Remote validation failed with exit code $exit_code"
    fi
    
    # Cleanup
    rm -f "${package_name}"
    
    return $exit_code
}

# Dry run mode
execute_dryrun() {
    log_step "Executing dry run (script validation only)..."
    
    echo ""
    log_info "Checking scripts..."
    
    local scripts=(
        "validate_h100_7loop.sh"
        "collect_logs.sh"
        "build_containers.sh"
        "quick_start.sh"
    )
    
    local all_ok=true
    
    for script in "${scripts[@]}"; do
        if [ -f "${SCRIPT_DIR}/${script}" ]; then
            if bash -n "${SCRIPT_DIR}/${script}" 2>/dev/null; then
                log_success "${script}: syntax OK"
            else
                log_error "${script}: syntax error"
                all_ok=false
            fi
        else
            log_warning "${script}: not found"
        fi
    done
    
    echo ""
    if [ "$all_ok" = true ]; then
        log_success "Dry run complete: All scripts valid"
        return 0
    else
        log_error "Dry run complete: Some scripts have issues"
        return 1
    fi
}

# Main execution
main() {
    print_banner
    
    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed. Aborting."
        exit 1
    fi
    
    echo ""
    
    # Get execution mode
    mode=$(select_execution_mode)
    
    log_info "Execution mode: ${mode}"
    echo ""
    
    # Execute based on mode
    case $mode in
        local)
            execute_local
            exit $?
            ;;
        remote)
            execute_remote
            exit $?
            ;;
        dryrun)
            execute_dryrun
            exit $?
            ;;
        *)
            log_error "Invalid mode: ${mode}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

