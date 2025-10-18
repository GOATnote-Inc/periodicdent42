#pragma once

// Minimal NVTX shim for profiling ranges
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_RANGE_PUSH(name) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#endif

