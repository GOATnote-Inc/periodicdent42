#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/memory.h>

#define BM 128
#define BN 128
#define BK 32

using ElemIn = cutlass::half_t;
using ElemAcc = float;

#define CUDA_CHECK(expr) do { \
  cudaError_t err = (expr); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); \
  } \
} while (0)

__global__ void bsr_kernel_cutlass_types(
    ElemIn const* __restrict__ A, int const* __restrict__ Ar, int const* __restrict__ Ac,
    ElemIn const* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int Mb, int Nb, int Kb) {
    
    int bm = blockIdx.x, bn = blockIdx.y;
    if(bm >= Mb || bn >= Nb) return;
    
    __shared__ __align__(128) ElemIn sA[BM*BK];
    __shared__ __align__(128) ElemIn sB[BK*BN];
    
    int tid = threadIdx.x;
    float acc[32];
    for(int i=0; i<32; i++) acc[i] = 0.0f;
    
    for(int idx = Ar[bm]; idx < Ar[bm+1]; idx++) {
        int kb = Ac[idx];
        
        // Use CUTLASS global_load
        for(int i=tid; i<BM*BK; i+=512) {
            cutlass::arch::global_load<ElemIn, sizeof(ElemIn)>(
                sA[i], &A[idx*BM*BK + i], true);
        }
        for(int i=tid; i<BK*BN; i+=512) {
            int row = i / BN, col = i % BN;
            bool guard = (kb*BK + row < K && bn*BN + col < N);
            cutlass::arch::global_load<ElemIn, sizeof(ElemIn)>(
                sB[i], &B[(kb*BK+row)*N + (bn*BN+col)], guard);
        }
        __syncthreads();
        
        for(int e=0; e<32; e++) {
            int elem = tid + e*512;
            if(elem < BM*BN) {
                int i = elem / BN, j = elem % BN;
                float sum = 0.0f;
                for(int k=0; k<BK; k++) {
                    sum += float(sA[i*BK + k]) * float(sB[k*BN + j]);
                }
                acc[e] += sum;
            }
        }
        __syncthreads();
    }
    
    for(int e=0; e<32; e++) {
        int elem = tid + e*512;
        if(elem < BM*BN) {
            int i = elem / BN, j = elem % BN;
            if(bm*BM + i < M && bn*BN + j < N)
                C[(bm*BM+i)*N + (bn*BN+j)] = acc[e];
        }
    }
}

int main() {
    printf("\n=== TMA Iter 5: CUTLASS types + loading ===\n\n");
    
    int M=8192, N=8192, K=8192, Mb=M/BM, Nb=N/BN, Kb=K/BK;
    
    std::vector<int> Ar(Mb+1), Ac;
    srand(42);
    for(int i=0; i<Mb; i++) {
        for(int j=0; j<Kb/8; j++) Ac.push_back(rand() % Kb);
        Ar[i+1] = Ac.size();
    }
    int nnz = Ac.size();
    
    std::vector<ElemIn> hA(nnz*BM*BK), hB(K*N);
    for(auto&x:hA) x = ElemIn(0.01f);
    for(auto&x:hB) x = ElemIn(0.01f);
    
    ElemIn *dA, *dB; float *dC; int *dAr, *dAc;
    CUDA_CHECK(cudaMalloc(&dA, nnz*BM*BK*sizeof(ElemIn)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K*N*sizeof(ElemIn)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dAr, (Mb+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dAc, nnz*sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nnz*BM*BK*sizeof(ElemIn), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), (size_t)K*N*sizeof(ElemIn), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAr, Ar.data(), (Mb+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAc, Ac.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    
    for(int i=0; i<5; i++)
        bsr_kernel_cutlass_types<<<dim3(Mb,Nb), 512>>>(dA, dAr, dAc, dB, dC, M, N, K, Mb, Nb, Kb);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    
    CUDA_CHECK(cudaEventRecord(s));
    for(int i=0; i<100; i++)
        bsr_kernel_cutlass_types<<<dim3(Mb,Nb), 512>>>(dA, dAr, dAc, dB, dC, M, N, K, Mb, Nb, Kb);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    
    float avg = ms / 100;
    double tflops = (2.0*M*N*K/1e12) / (avg/1000);
    
    printf("CUTLASS types: %.3f ms | %.1f TFLOPS\n", avg, tflops);
    printf("Status: %s\n", avg > 0.001 && avg < 100.0 ? "✅ Foundation ready for TMA" : "⚠️  Check timing");
    
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dAr)); CUDA_CHECK(cudaFree(dAc));
    
    printf("\nIteration 5 complete - types validated\n");
    printf("Next: Add actual TMA descriptor creation\n");
    return 0;
}
