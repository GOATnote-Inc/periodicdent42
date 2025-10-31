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

.PHONY: all clean kernel verify profile baseline

all: kernel

kernel:
	@mkdir -p build
	@echo "Building $(TARGET) (TMA=$(TMA))..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)
	@echo "✅ Built: $(TARGET)"

verify: kernel
	@echo "Running verification..."
	@$(TARGET)

profile: kernel
	@echo "Profiling $(TARGET)..."
	@ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
		--target-processes all $(TARGET) 2>&1 | tee artifacts/profile.log

baseline: kernel
	@echo "Establishing baseline metrics..."
	@python3 scripts/extract_baseline.py --output ci/baseline/nsight_baseline.norm.txt

clean:
	rm -rf build/*.o build/sparse_h100* artifacts/*.log artifacts/*.csv artifacts/*.ncu-rep

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
