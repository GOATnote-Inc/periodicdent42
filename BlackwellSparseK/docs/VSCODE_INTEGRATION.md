# VS Code Integration Guide

**BlackwellSparseK v0.1.0** - Production-Grade IDE Integration

---

## 🎯 Quick Start

Open BlackwellSparseK in VS Code, then press:

- **`Ctrl+Shift+B`** (Windows/Linux) or **`Cmd+Shift+B`** (macOS) → Run default task (H100 Validation)
- **`Ctrl+Shift+P`** → Type "Tasks: Run Task" → Select any BlackwellSparseK task

---

## 📋 Available Tasks

### 1. **H100 Validation (Default Build Task)**
```
Task: BlackwellSparseK: H100 Validation
Shortcut: Ctrl+Shift+B (Cmd+Shift+B on macOS)
```

**What it does:**
- Interactive orchestrator with 3 execution modes:
  1. **Local H100**: Direct execution on local GPU
  2. **Remote H100**: SSH to RunPod/Vast.ai instance
  3. **Dry Run**: Validate scripts without execution

**Output:**
- Real-time progress with color-coded logs
- Dedicated terminal panel with full output
- Validation report at `/workspace/results/H100_VALIDATION_REPORT.md`

**When to use:**
- Before committing kernel changes
- After modifying CUDA kernels
- For performance regression testing
- Before production deployment

---

### 2. **Build Containers**
```
Task: BlackwellSparseK: Build Containers
```

**What it does:**
- Builds all 4 Docker images (dev, prod, bench, ci)
- Tags with version and latest
- Validates multi-stage builds

**Duration:** ~20-30 minutes (first run), ~5 minutes (cached)

---

### 3. **Quick Start Dev**
```
Task: BlackwellSparseK: Quick Start Dev
```

**What it does:**
- Launches interactive development container
- Mounts workspace as volume
- Pre-configured with all dependencies
- Ideal for iterative kernel development

**Usage:**
```bash
# Inside container:
cd /workspace/BlackwellSparseK
python -c "import blackwell_sparsek; print('OK')"
```

---

### 4. **Run Tests (Docker)**
```
Task: BlackwellSparseK: Run Tests (Docker)
```

**What it does:**
- Runs pytest suite in CI container
- No GPU required (unit tests only)
- Fast feedback loop for Python code

**Coverage:**
- Unit tests for core modules
- Integration tests (non-GPU)
- Code coverage report

---

### 5. **Run Benchmarks**
```
Task: BlackwellSparseK: Run Benchmarks
```

**What it does:**
- Launches benchmark container with GPU
- Runs latency benchmarks
- Compares against PyTorch SDPA baseline
- Captures Nsight Compute metrics

**Output:**
- Results saved to `results/` directory
- CSV files for trend analysis
- Nsight Compute reports (.ncu-rep)

---

### 6. **Validate H100 (Remote)**
```
Task: BlackwellSparseK: Validate H100 (Remote)
```

**What it does:**
- Direct call to `validate_h100.sh`
- Automated SSH connection to RunPod
- Package upload and execution
- Result download

**Use case:**
- Automated CI/CD integration
- Scripted validation in pipelines

---

### 7. **Lint Code**
```
Task: BlackwellSparseK: Lint Code
```

**What it does:**
- Runs ruff on Python code
- Checks PEP 8 compliance
- Identifies common issues

**Problem Matcher:**
- Errors appear in VS Code Problems panel
- Click to jump to file/line

---

### 8. **Format Code**
```
Task: BlackwellSparseK: Format Code
```

**What it does:**
- Auto-formats Python code with black
- Applies consistent style
- Runs silently (format-on-save enabled)

---

### 9. **View Validation Report**
```
Task: BlackwellSparseK: View Validation Report
```

**What it does:**
- Opens latest H100 validation report
- New panel for focused reading
- Checks both local and container paths

---

### 10. **Clean Build Artifacts**
```
Task: BlackwellSparseK: Clean Build Artifacts
```

**What it does:**
- Removes compiled `.so` files
- Clears Python cache (`__pycache__`)
- Deletes build/dist directories

**When to use:**
- Before fresh build
- When resolving import issues
- After changing build configuration

---

## 🚀 Typical Development Workflows

### **Workflow 1: Kernel Development**
```
1. Edit kernel (src/blackwell_sparsek/kernels/*.cu)
2. Task: "Clean Build Artifacts"
3. Task: "Build Containers" (or local pip install -e .)
4. Task: "Run Tests (Docker)"
5. Task: "H100 Validation" (select Local or Remote)
6. Commit if validation passes
```

### **Workflow 2: Python Integration**
```
1. Edit Python code (src/blackwell_sparsek/backends/*.py)
2. Task: "Lint Code"
3. Task: "Format Code" (auto-applies on save)
4. Task: "Run Tests (Docker)"
5. Commit
```

### **Workflow 3: Performance Benchmarking**
```
1. Task: "Build Containers"
2. Task: "Run Benchmarks"
3. View results in results/*.csv
4. Task: "View Validation Report"
```

### **Workflow 4: Remote H100 Validation**
```
1. Start RunPod H100 instance
2. Note IP and port from RunPod dashboard
3. Task: "H100 Validation"
4. Select option 2 (Remote H100)
5. Enter IP and port when prompted
6. Wait for upload + validation
7. Report downloaded to results/
```

---

## 🛠️ Orchestrator Script Details

**Location:** `scripts/h100_orchestrator.sh`

**Features:**
- ✅ Interactive mode selection (local/remote/dry-run)
- ✅ SSH connection validation with retry logic
- ✅ Automatic package creation and upload
- ✅ Color-coded output for terminal visibility
- ✅ Progress indicators and status checks
- ✅ Result download and report generation
- ✅ Error handling with clear diagnostics

**Execution Modes:**

### **Mode 1: Local H100**
- Requires: Local H100 GPU, nvidia-smi, CUDA 13.0+
- Runs: Direct validation on local machine
- Output: Real-time terminal feedback
- Report: Saved to results/H100_VALIDATION_REPORT.md

### **Mode 2: Remote H100**
- Requires: SSH access to RunPod/Vast.ai, SSH key configured
- Process:
  1. Test SSH connection
  2. Create deployment tarball
  3. Upload to /workspace
  4. Extract and execute validation
  5. Download results
- Output: SSH output streamed to terminal
- Report: Downloaded from remote to local results/

### **Mode 3: Dry Run**
- Requires: None (CPU only)
- Validates: Shell script syntax
- Output: List of scripts with validation status
- Use: Quick sanity check before remote execution

---

## 🐛 Debugging with VS Code

### **Python Debugging**
```
1. Open benchmarks/perf.py
2. Set breakpoint (click line number)
3. F5 → "Python: Benchmarks"
4. Step through code with F10/F11
```

### **CUDA Debugging (Advanced)**
```
Prerequisites:
- cuda-gdb installed (/usr/local/cuda/bin/cuda-gdb)
- Debug build of kernels (cmake -DCMAKE_BUILD_TYPE=Debug)

Steps:
1. Launch Python script with CUDA kernel call
2. Get process ID: ps aux | grep python
3. F5 → "CUDA-GDB: Attach"
4. Enter process ID when prompted
5. Set breakpoint in .cu file
6. Use cuda-gdb commands:
   - info cuda threads
   - cuda thread (0,0,0)
   - print var_name
```

---

## ⚙️ Configuration Files

### **tasks.json**
Defines all available tasks and their configurations.

**Key fields:**
- `command`: Shell command to execute
- `group`: Task category (build/test/none)
- `presentation`: Terminal panel behavior
- `problemMatcher`: Error parsing for Problems panel

### **settings.json**
IDE-level settings for BlackwellSparseK.

**Includes:**
- CUDA file associations (*.cu → cuda-cpp)
- C++ include paths (CUDA 13.0, CUTLASS 4.3.0)
- Python linting (ruff) and formatting (black)
- Test framework (pytest)
- Format-on-save enabled
- Terminal environment variables (CUDA_HOME, CUTLASS_PATH)

### **launch.json**
Debugger configurations for Python and CUDA.

**Configurations:**
- Python: Current File (generic)
- Python: Benchmarks (pre-configured args)
- Python: Test Current File (pytest single file)
- Python: All Tests (full suite)
- Python: GPU Tests Only (pytest -m gpu)
- CUDA-GDB: Attach (kernel debugging)

---

## 📦 Integration with CI/CD

VS Code tasks can be invoked from GitHub Actions:

```yaml
# .github/workflows/ci.yml
jobs:
  vscode-integration-test:
    runs-on: [self-hosted, gpu, h100]
    steps:
      - uses: actions/checkout@v4
      - name: Run H100 validation via orchestrator
        run: |
          cd BlackwellSparseK
          bash scripts/h100_orchestrator.sh <<< "1"  # Select local mode
```

---

## 🎨 Terminal Output Example

```
==========================================
  BlackwellSparseK H100 Validation
==========================================
  Version: 0.1.0
  Timestamp: 2025-10-30_15-30-45
  Mode: VS Code Integrated
==========================================

[15:30:45] ▶ Checking prerequisites...
[15:30:45] ✅ Prerequisites check passed

▶ Select execution mode:

  1) Local H100 (direct execution)
  2) Remote H100 (SSH to RunPod/Vast.ai)
  3) Dry run (check scripts only)

Enter choice [1-3]: 1

[15:30:47] ℹ️  Execution mode: local

[15:30:47] ▶ Executing local H100 validation...
[15:30:47] ℹ️  Starting 7-loop validation framework...

--- LOOP 1: ANALYZE ---
[15:30:48] GPU: NVIDIA H100 80GB HBM3
[15:30:48] CUDA: 13.0.2
[15:30:48] PyTorch: 2.9.0+cu130
...
```

---

## 🔧 Troubleshooting

### **Issue: Task not found**
**Solution:**
```bash
# Reload VS Code window
Ctrl+Shift+P → "Reload Window"
```

### **Issue: Docker commands fail**
**Solution:**
```bash
# Check Docker daemon
docker info

# Ensure Docker Compose installed
docker-compose --version
```

### **Issue: SSH to RunPod fails**
**Solution:**
```bash
# Test SSH manually
ssh -p <PORT> root@<IP> "echo test"

# Check SSH key
cat ~/.ssh/id_rsa.pub
```

### **Issue: CUDA not detected in container**
**Solution:**
```bash
# Ensure nvidia-docker runtime
docker run --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi
```

---

## 📚 Additional Resources

- **Project README**: `../README.md`
- **Architecture Docs**: `ARCHITECTURE.md`
- **API Reference**: `API_REFERENCE.md`
- **Migration Guide**: `MIGRATION_FROM_FLASHCORE.md`

---

## 🎓 Pro Tips

1. **Parallel Tasks**: Tasks run independently, can execute multiple simultaneously
2. **Terminal History**: Use `Ctrl+Up/Down` to navigate previous task outputs
3. **Quick Task Repeat**: Press `Ctrl+Shift+B` twice to repeat last build task
4. **Problems Panel**: `Ctrl+Shift+M` to view all linting errors
5. **Integrated Terminal**: `Ctrl+` ` to toggle terminal panel
6. **Task Output Filtering**: Click terminal dropdown to switch between task outputs

---

**Created**: 2025-10-30  
**Author**: BlackwellSparseK Development Team  
**License**: Apache 2.0  

