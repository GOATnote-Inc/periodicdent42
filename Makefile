# Makefile for rapid H100 iteration
NVCC = nvcc
ARCH = sm_90a
FLAGS = -arch=$(ARCH) -O3 --use_fast_math -lineinfo --ptxas-options=-v -I.

all: step1 step2 step3

step1:
	$(NVCC) $(FLAGS) test_wgmma_single_corrected.cu -o build/test_step1

step2:
	$(NVCC) $(FLAGS) -DSTEP=2 test_wgmma_single_corrected.cu -o build/test_step2

step3:
	$(NVCC) $(FLAGS) -DSTEP=3 test_wgmma_single_corrected.cu -o build/test_step3

clean:
	rm -f build/test_step*

test: step1
	./build/test_step1 | grep -E "TFLOPS|Status"

profile: step1
	ncu --set full ./build/test_step1

.PHONY: all clean test profile
