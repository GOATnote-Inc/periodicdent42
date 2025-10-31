# BlackwellSparseK Makefile
# CUDA 13.0.2 + CUTLASS 4.3.0 + H100 (sm_90a)

CUDA_HOME ?= /usr/local/cuda-13.0
CUTLASS_PATH ?= /opt/cutlass

CXX := $(CUDA_HOME)/bin/nvcc
CXXFLAGS := -O3 -std=c++17 -arch=sm_90a -lineinfo -Xptxas=-v
INCLUDES := -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include
LDFLAGS := -L$(CUDA_HOME)/lib64

# TMA toggle: make kernel TMA=1
TMA ?= 0

ifeq ($(TMA),1)
SRC := src/experimental/tma/bsr_gemm_tma.cu
TARGET := build/sparse_h100_tma
else
SRC := src/sparse_bsr_gemm_h100.cu
TARGET := build/sparse_h100
endif

.PHONY: all clean kernel verify bench bench-full baseline container

all: kernel

kernel:
	@mkdir -p build
	@echo "Building $(TARGET) (TMA=$(TMA))..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)
	@echo "✅ Built: $(TARGET)"

verify: kernel
	@echo "Running verification..."
	@$(TARGET)

# Quick benchmark (10 trials, 30s warmup)
bench: kernel
	@echo "Quick benchmark..."
	@TRIALS=10 WARMUP=30 BINARY=$(TARGET) bash scripts/bench.sh

# Full benchmark (30 trials, 60s warmup, Nsight profiling)
bench-full: kernel
	@echo "Full benchmark with profiling..."
	@TRIALS=30 WARMUP=60 PROFILE=1 BINARY=$(TARGET) bash scripts/bench.sh

baseline: kernel
	@echo "Establishing baseline metrics..."
	@TRIALS=30 WARMUP=60 PROFILE=1 BINARY=$(TARGET) bash scripts/bench.sh
	@python3 scripts/extract_baseline.py \
		--log artifacts/ncu.log \
		--output ci/baseline/nsight_baseline.norm.txt
	@echo "✅ Baseline established: ci/baseline/nsight_baseline.norm.txt"
	@cat ci/baseline/nsight_baseline.norm.txt

# Build profiling container
container:
	@echo "Building profiling container..."
	docker build -f Dockerfile.profiling -t blackwellsparsek:profiling .
	@echo "✅ Container ready: blackwellsparsek:profiling"

clean:
	rm -rf build/*.o build/sparse_h100* artifacts/*.log artifacts/*.csv artifacts/*.ncu-rep artifacts/*.txt

help:
	@echo "BlackwellSparseK Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make kernel         - Build baseline kernel (TMA=0)"
	@echo "  make kernel TMA=1   - Build TMA experimental kernel"
	@echo "  make verify         - Build and run"
	@echo "  make profile        - Profile with Nsight Compute"
	@echo "  make baseline       - Extract baseline metrics"
	@echo "  make clean          - Clean build artifacts"
